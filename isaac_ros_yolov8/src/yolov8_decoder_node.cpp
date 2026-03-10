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

#include "isaac_ros_yolov8/yolov8_decoder_node.hpp"

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
namespace yolov8
{
YoloV8DecoderNode::YoloV8DecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov8_decoder_node", options),
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "tensor_sub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&YoloV8DecoderNode::InputCallback, this,
      std::placeholders::_1))},
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "detections_output", 50)},
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
  num_classes_{declare_parameter<int64_t>("num_classes", 80)}
{}

YoloV8DecoderNode::~YoloV8DecoderNode() = default;
void YoloV8DecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  auto tensor = msg.GetNamedTensor(tensor_name_);
  size_t buffer_size{tensor.GetTensorSize()};
  size_t num_elements = buffer_size / sizeof(float);

  // Validate tensor size
  if (num_elements != (1 * 5 * 8400)) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Unexpected tensor size. Expected %lu elements but got %lu",
      static_cast<size_t>(1 * 5 * 8400),
      num_elements);
    return;
  }

  std::vector<float> results_vector{};
  results_vector.resize(buffer_size);
  cudaMemcpy(results_vector.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> indices;
  std::vector<int> classes;

  // Output dimensions = [1, 5, 8400]
  int out_dim = 8400;
  float * results_data = reinterpret_cast<float *>(results_vector.data());

  for (int i = 0; i < out_dim; i++) {
    float x = *(results_data + i);
    float y = *(results_data + (out_dim * 1) + i);
    float w = *(results_data + (out_dim * 2) + i);
    float h = *(results_data + (out_dim * 3) + i);
    float score = *(results_data + (out_dim * 4) + i);

    float x1 = x - (0.5 * w);
    float y1 = y - (0.5 * h);

    // Filter low confidence boxes early
    if (score < confidence_threshold_) {
      continue;
    }

    bboxes.emplace_back(x1, y1, w, h);
    scores.emplace_back(score);

    // Only one class exists
    classes.emplace_back(0);
  }

  RCLCPP_DEBUG(this->get_logger(), "Count of bboxes: %lu", bboxes.size());

  cv::dnn::NMSBoxes(
    bboxes,
    scores,
    confidence_threshold_,
    nms_threshold_,
    indices,
    5);

  RCLCPP_DEBUG(this->get_logger(), "# boxes after NMS: %lu", indices.size());

  vision_msgs::msg::Detection2DArray final_detections_arr;

  for (size_t i = 0; i < indices.size(); i++) {
    int ind = indices[i];
    vision_msgs::msg::Detection2D detection;

    // 2D object Bbox
    float w = bboxes[ind].width;
    float h = bboxes[ind].height;
    float x_center = bboxes[ind].x + (0.5 * w);
    float y_center = bboxes[ind].y + (0.5 * h);

    detection.bbox.center.position.x = x_center;
    detection.bbox.center.position.y = y_center;
    detection.bbox.size_x = w;
    detection.bbox.size_y = h;

    // Single class hypothesis
    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = "0";
    hyp.hypothesis.score = scores[ind];

    detection.results.push_back(hyp);

    detection.header.stamp.sec = msg.GetTimestampSeconds();
    detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();

    final_detections_arr.detections.push_back(detection);
  }

  final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
  final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();

  pub_->publish(final_detections_arr);
}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::YoloV8DecoderNode)
