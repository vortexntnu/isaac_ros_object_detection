// SPDX-License-Identifier: MIT

#include "isaac_ros_yolov26_obb/yolov26_obb_decoder_node.hpp"

#include <cuda_runtime.h>
#include <vector>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

#include "vision_msgs/msg/detection2_d_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov26_obb
{

// OBB output tensor layout: [1, num_detections, VALUES_PER_DET]
// Per-detection values: [cx, cy, w, h, angle_rad, confidence, class_id]
static constexpr int VALUES_PER_DET = 7;

YoloV26OBBDecoderNode::YoloV26OBBDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov26_obb_decoder_node", options),
  tensor_input_topic_{declare_parameter<std::string>("tensor_input_topic", "tensor_sub")},
  detections_topic_{declare_parameter<std::string>("detections_topic", "detections_output")},
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  num_detections_{static_cast<int>(declare_parameter<int64_t>("num_detections", 300))},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      tensor_input_topic_,
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&YoloV26OBBDecoderNode::InputCallback, this, std::placeholders::_1))},
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(detections_topic_, 50)}
{}

YoloV26OBBDecoderNode::~YoloV26OBBDecoderNode() = default;

void YoloV26OBBDecoderNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  auto tensor = msg.GetNamedTensor(tensor_name_);
  const size_t buffer_size = tensor.GetTensorSize();
  const size_t num_elements = buffer_size / sizeof(float);
  const size_t expected_elements = static_cast<size_t>(num_detections_ * VALUES_PER_DET);

  if (num_elements != expected_elements) {
    RCLCPP_ERROR(
      get_logger(),
      "Unexpected tensor size. Expected %zu elements [%d * %d] but got %zu. "
      "Check num_detections parameter and that you are using a YOLO-OBB model.",
      expected_elements, num_detections_, VALUES_PER_DET, num_elements);
    return;
  }

  std::vector<float> data(num_elements);
  cudaMemcpy(data.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);

  vision_msgs::msg::Detection2DArray out;
  out.header.stamp.sec = msg.GetTimestampSeconds();
  out.header.stamp.nanosec = msg.GetTimestampNanoseconds();

  for (int i = 0; i < num_detections_; ++i) {
    const float * det = data.data() + i * VALUES_PER_DET;

    // Per-detection layout: [cx, cy, w, h, confidence, class_id, angle_rad]
    const float cx    = det[0];
    const float cy    = det[1];
    const float w     = det[2];
    const float h     = det[3];
    const float conf  = det[4];
    const float cls   = det[5];
    const float angle = det[6];  // radians, matches BoundingBox2D.center.theta

    if (conf < static_cast<float>(confidence_threshold_)) {
      continue;
    }

    vision_msgs::msg::Detection2D detection;
    detection.header = out.header;

    detection.bbox.center.position.x = cx;
    detection.bbox.center.position.y = cy;
    detection.bbox.center.theta = angle;
    detection.bbox.size_x = w;
    detection.bbox.size_y = h;

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(static_cast<int>(cls));
    hyp.hypothesis.score = conf;
    detection.results.push_back(hyp);

    out.detections.push_back(detection);
  }

  RCLCPP_DEBUG(get_logger(), "Published %zu OBB detections", out.detections.size());
  pub_->publish(out);
}

}  // namespace yolov26_obb
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov26_obb::YoloV26OBBDecoderNode)
