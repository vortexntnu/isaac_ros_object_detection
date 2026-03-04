// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_yolov26/yolov26_decoder_node.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

#include "vision_msgs/msg/detection2_d_array.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov26
{

YoloV26DecoderNode::YoloV26DecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov26_decoder_node", options),
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "tensor_sub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&YoloV26DecoderNode::InputCallback, this,
      std::placeholders::_1))},
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "detections_output", 50)},
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
  num_classes_{declare_parameter<int64_t>("num_classes", 80)}
{}

YoloV26DecoderNode::~YoloV26DecoderNode() = default;

void YoloV26DecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  auto tensor = msg.GetNamedTensor(tensor_name_);
  size_t buffer_size{tensor.GetTensorSize()};

  // NOTE: This matches the 3.2-style YOLOv8 decoder pattern you provided.
  // buffer_size is bytes; we allocate same-sized float vector and reinterpret,
  // identical to upstream pattern.
  std::vector<float> results_vector{};
  results_vector.resize(buffer_size);
  cudaMemcpy(results_vector.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);

  // We support two layouts:
  //  - axis-aligned: [1, 4+num_classes, out_dim]
  //  - OBB:          [1, 5+num_classes, out_dim] (theta in radians)
  //
  // Per your instruction: assume same shapes as YOLOv8 (out_dim=8400) unless you change later.
  int out_dim = 8400;

  float * results_data = reinterpret_cast<float *>(results_vector.data());

  // Determine whether theta exists by assuming channels match either 4+num_classes or 5+num_classes.
  // With the 3.2 code style, we don't read tensor shape metadata, so this is a best-effort guess:
  // If you export OBB, you should set num_classes properly and ensure the engine output includes theta.
  const bool has_theta = true;  // Default to true for YOLOv26 OBB use-case.

  std::vector<cv::Rect> aabbs;
  std::vector<cv::RotatedRect> rbbs;
  std::vector<float> scores;
  std::vector<int> indices;
  std::vector<int> classes;

  aabbs.reserve(out_dim);
  rbbs.reserve(out_dim);
  scores.reserve(out_dim);
  indices.reserve(out_dim);
  classes.reserve(out_dim);

  for (int i = 0; i < out_dim; i++) {
    float cx = *(results_data + (out_dim * 0) + i);
    float cy = *(results_data + (out_dim * 1) + i);
    float w  = *(results_data + (out_dim * 2) + i);
    float h  = *(results_data + (out_dim * 3) + i);

    float theta = 0.0f;
    int class_offset = 4;

    if (has_theta) {
      theta = *(results_data + (out_dim * 4) + i);  // radians
      class_offset = 5;
    }

    std::vector<float> conf;
    conf.reserve(static_cast<size_t>(num_classes_));

    for (int j = 0; j < num_classes_; j++) {
      conf.push_back(*(results_data + (out_dim * (class_offset + j)) + i));
    }

    std::vector<float>::iterator ind_max_conf;
    ind_max_conf = std::max_element(std::begin(conf), std::end(conf));
    int max_index = distance(std::begin(conf), ind_max_conf);
    float val_max_conf = *max_element(std::begin(conf), std::end(conf));

    scores.push_back(val_max_conf);
    classes.push_back(max_index);
    indices.push_back(i);

    // Build axis-aligned box (for fallback / debugging)
    float x1 = (cx - (0.5f * w));
    float y1 = (cy - (0.5f * h));
    aabbs.push_back(cv::Rect(x1, y1, w, h));

    // Build rotated box for OBB NMS/drawing (angle in degrees for OpenCV)
    float theta_deg = theta * 180.0f / static_cast<float>(M_PI);
    rbbs.push_back(cv::RotatedRect(cv::Point2f(cx, cy), cv::Size2f(w, h), theta_deg));
  }

  RCLCPP_DEBUG(this->get_logger(), "Count of bboxes: %lu", rbbs.size());

  // Rotated NMS (OBB). If your model is not OBB, set has_theta=false above.
  cv::dnn::NMSBoxesRotated(
    rbbs, scores, static_cast<float>(confidence_threshold_),
    static_cast<float>(nms_threshold_), indices, 5);

  RCLCPP_DEBUG(this->get_logger(), "# boxes after NMS: %lu", indices.size());

  vision_msgs::msg::Detection2DArray final_detections_arr;

  for (size_t i = 0; i < indices.size(); i++) {
    int ind = indices[i];
    vision_msgs::msg::Detection2D detection;

    // 2D object Bbox (center + theta + size)
    const cv::RotatedRect & rr = rbbs[ind];
    detection.bbox.center.position.x = rr.center.x;
    detection.bbox.center.position.y = rr.center.y;
    detection.bbox.size_x = rr.size.width;
    detection.bbox.size_y = rr.size.height;

    // BoundingBox2D.center.theta uses radians; OpenCV RotatedRect uses degrees.
    detection.bbox.center.theta = static_cast<double>(rr.angle) * M_PI / 180.0;

    // Class probabilities
    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(classes.at(ind));
    hyp.hypothesis.score = scores.at(ind);
    detection.results.push_back(hyp);

    detection.header.stamp.sec = msg.GetTimestampSeconds();
    detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();

    final_detections_arr.detections.push_back(detection);
  }

  final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
  final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  pub_->publish(final_detections_arr);
}

}  // namespace yolov26
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov26::YoloV26DecoderNode)
