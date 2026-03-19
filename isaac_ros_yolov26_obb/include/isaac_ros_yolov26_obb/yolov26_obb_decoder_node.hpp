// SPDX-License-Identifier: MIT

#ifndef ISAAC_ROS_YOLOV26_OBB__YOLOV26_OBB_DECODER_NODE_HPP_
#define ISAAC_ROS_YOLOV26_OBB__YOLOV26_OBB_DECODER_NODE_HPP_

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"

#include "vision_msgs/msg/detection2_d_array.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov26_obb
{

// Decodes YOLO-OBB TensorRT output tensor [1, 300, 7] into Detection2DArray.
//
// Expected per-detection layout (7 floats):
//   [cx, cy, w, h, confidence, class_id, angle_rad]
//
// NMS is assumed to be embedded in the model export (no NMS applied here).
class YoloV26OBBDecoderNode : public rclcpp::Node
{
public:
  explicit YoloV26OBBDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~YoloV26OBBDecoderNode();

private:
  void InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg);

  // Topic names
  std::string tensor_input_topic_{};
  std::string detections_topic_{};

  // Name of tensor in NitrosTensorList
  std::string tensor_name_{};

  // Number of detection slots in the tensor (inner dim 1)
  int num_detections_{};

  // Confidence threshold for filtering padded / low-confidence slots
  double confidence_threshold_{};

  // Subscription to NitrosTensorList messages
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>> nitros_sub_;

  // Publisher for Detection2DArray
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;
};

}  // namespace yolov26_obb
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV26_OBB__YOLOV26_OBB_DECODER_NODE_HPP_
