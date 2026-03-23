// SPDX-License-Identifier: MIT

#include "isaac_ros_yolov26_cls/yolov26_cls_decoder_node.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

#include "std_msgs/msg/u_int8.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov26_cls
{

YoloV26ClsDecoderNode::YoloV26ClsDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov26_cls_decoder_node", options),
  tensor_input_topic_{declare_parameter<std::string>("tensor_input_topic", "tensor_sub")},
  class_topic_{declare_parameter<std::string>("class_topic", "classification_output")},
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  num_classes_{static_cast<int>(declare_parameter<int64_t>("num_classes", 10))},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      tensor_input_topic_,
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&YoloV26ClsDecoderNode::InputCallback, this, std::placeholders::_1))},
  pub_{create_publisher<std_msgs::msg::UInt8>(class_topic_, 50)}
{}

YoloV26ClsDecoderNode::~YoloV26ClsDecoderNode() = default;

void YoloV26ClsDecoderNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  auto tensor = msg.GetNamedTensor(tensor_name_);
  const size_t buffer_size = tensor.GetTensorSize();
  const size_t num_elements = buffer_size / sizeof(float);

  if (static_cast<int>(num_elements) != num_classes_) {
    RCLCPP_ERROR(
      get_logger(),
      "Unexpected tensor size. Expected %d class scores but got %zu elements. "
      "Check num_classes parameter and that you are using a YOLO classification model.",
      num_classes_, num_elements);
    return;
  }

  std::vector<float> scores(num_elements);
  cudaMemcpy(scores.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);

  // Find the class with the highest score (argmax)
  auto max_it = std::max_element(scores.begin(), scores.end());
  int class_id = static_cast<int>(std::distance(scores.begin(), max_it));
  float max_score = *max_it;

  if (max_score < static_cast<float>(confidence_threshold_)) {
    RCLCPP_DEBUG(
      get_logger(),
      "Top class %d score %.3f below threshold %.3f, skipping",
      class_id, max_score, confidence_threshold_);
    return;
  }

  std_msgs::msg::UInt8 out;
  out.data = static_cast<uint8_t>(class_id);
  pub_->publish(out);

  RCLCPP_DEBUG(
    get_logger(),
    "Published classification: class %d (score %.3f)", class_id, max_score);
}

}  // namespace yolov26_cls
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov26_cls::YoloV26ClsDecoderNode)
