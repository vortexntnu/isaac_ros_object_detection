// SPDX-License-Identifier: MIT

#ifndef ISAAC_ROS_YOLOV26_CLS__YOLOV26_CLS_DECODER_NODE_HPP_
#define ISAAC_ROS_YOLOV26_CLS__YOLOV26_CLS_DECODER_NODE_HPP_

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"

#include "std_msgs/msg/u_int8.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov26_cls
{

// Decodes YOLO classification TensorRT output tensor [1, num_classes] into a
// UInt8 message containing the predicted class index.
//
// The tensor contains raw logits or softmax probabilities for each class.
// The decoder applies argmax to find the top-1 class and publishes its index.
class YoloV26ClsDecoderNode : public rclcpp::Node
{
public:
  explicit YoloV26ClsDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~YoloV26ClsDecoderNode();

private:
  void InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg);

  // Topic names
  std::string tensor_input_topic_{};
  std::string class_topic_{};

  // Name of tensor in NitrosTensorList
  std::string tensor_name_{};

  // Number of classes in the classification model output
  int num_classes_{};

  // Confidence threshold (applied to softmax / max score)
  double confidence_threshold_{};

  // Subscription to NitrosTensorList messages
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>> nitros_sub_;

  // Publisher for predicted class ID
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr pub_;
};

}  // namespace yolov26_cls
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV26_CLS__YOLOV26_CLS_DECODER_NODE_HPP_
