#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Visualizer for YOLO classification results.

Subscribes to a UInt8 (class ID) and an Image, draws the classification label
(class number + configurable text) on the image, and republishes it.
"""

import ast

import cv2
import cv_bridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8


class YoloClsVisualizer(Node):
    QUEUE_SIZE = 10
    TXT_COLOR = (0, 255, 0)
    BG_COLOR = (0, 0, 0)

    def __init__(self):
        super().__init__('yolo_cls_visualizer')

        self.declare_parameter('class_topic', 'classification_output')
        self.declare_parameter('image_topic', '/yolo_cls_encoder/internal/resize/image')
        self.declare_parameter('output_image_topic', 'yolo_cls_processed_image')
        self.declare_parameter('class_names_yaml', "{0: 'object'}")

        class_topic = self.get_parameter('class_topic').value
        image_topic = self.get_parameter('image_topic').value
        output_image_topic = self.get_parameter('output_image_topic').value
        param = self.get_parameter('class_names_yaml').value

        self._bridge = cv_bridge.CvBridge()
        self._pub = self.create_publisher(Image, output_image_topic, self.QUEUE_SIZE)

        self._latest_class_id = None

        self.create_subscription(UInt8, class_topic, self._class_callback, self.QUEUE_SIZE)
        self.create_subscription(Image, image_topic, self._image_callback, self.QUEUE_SIZE)

        self.names = ast.literal_eval(param) if isinstance(param, str) else param

    def _class_callback(self, msg: UInt8):
        self._latest_class_id = msg.data

    def _image_callback(self, img_msg: Image):
        if self._latest_class_id is None:
            return

        img = self._bridge.imgmsg_to_cv2(img_msg)
        h, w = img.shape[:2]

        class_id = self._latest_class_id
        label = self.names.get(class_id, str(class_id))
        text = f'Class {class_id}: {label}'

        # Scale font to image size
        lw = max(round((h + w) / 2 * 0.003), 2)
        font_scale = lw / 3
        tf = max(lw - 1, 1)

        # Compute text size for background rectangle
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, tf)

        # Draw background rectangle at top-left
        margin = 10
        cv2.rectangle(
            img,
            (margin, margin),
            (margin + tw + 10, margin + th + baseline + 10),
            self.BG_COLOR, -1)

        # Draw text
        cv2.putText(
            img, text,
            (margin + 5, margin + th + 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            self.TXT_COLOR, thickness=tf, lineType=cv2.LINE_AA)

        out_msg = self._bridge.cv2_to_imgmsg(img, encoding=img_msg.encoding)
        out_msg.header = img_msg.header
        self._pub.publish(out_msg)


def main():
    rclpy.init()
    rclpy.spin(YoloClsVisualizer())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
