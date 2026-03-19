#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Visualizer for YOLO-OBB detections.

Subscribes to a Detection2DArray and an Image, draws oriented bounding boxes
(using BoundingBox2D.center.theta) on the image, and republishes it.
"""

import ast

import cv2
import cv_bridge
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class YoloOBBVisualizer(Node):
    QUEUE_SIZE = 10
    BOX_COLOR = (0, 255, 0)
    TXT_COLOR = (255, 0, 255)
    THICKNESS = 2

    def __init__(self):
        super().__init__('yolo_obb_visualizer')

        self.declare_parameter('detections_topic', 'detections_output')
        self.declare_parameter('image_topic', '/yolo_obb_encoder/internal/resize/image')
        self.declare_parameter('output_image_topic', 'yolo_obb_processed_image')
        self.declare_parameter('class_names_yaml', "{0: 'object'}")

        detections_topic = self.get_parameter('detections_topic').value
        image_topic = self.get_parameter('image_topic').value
        output_image_topic = self.get_parameter('output_image_topic').value
        param = self.get_parameter('class_names_yaml').value

        self._bridge = cv_bridge.CvBridge()
        self._pub = self.create_publisher(Image, output_image_topic, self.QUEUE_SIZE)

        self._det_sub = message_filters.Subscriber(self, Detection2DArray, detections_topic)
        self._img_sub = message_filters.Subscriber(self, Image, image_topic)

        self._sync = message_filters.TimeSynchronizer(
            [self._det_sub, self._img_sub], self.QUEUE_SIZE)
        self._sync.registerCallback(self._callback)

        self.names = ast.literal_eval(param) if isinstance(param, str) else param

    def _callback(self, detections_msg: Detection2DArray, img_msg: Image):
        img = self._bridge.imgmsg_to_cv2(img_msg)

        for det in detections_msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            theta_rad = det.bbox.center.theta  # radians

            class_id = int(det.results[0].hypothesis.class_id)
            score = det.results[0].hypothesis.score
            label = self.names.get(class_id, str(class_id))
            label = f'{label} {score:.2f}'

            # Build rotated rectangle and compute the 4 corner points
            rect = ((float(cx), float(cy)), (float(w), float(h)),
                    float(np.degrees(theta_rad)))
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            cv2.polylines(img, [box], isClosed=True,
                          color=self.BOX_COLOR, thickness=self.THICKNESS)

            # Label above the first corner
            lw = max(round((img_msg.height + img_msg.width) / 2 * 0.003), 2)
            tf = max(lw - 1, 1)
            x0, y0 = int(box[0][0]), int(box[0][1])
            cv2.putText(img, label, (x0, max(y0 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, lw / 3,
                        self.TXT_COLOR, thickness=tf, lineType=cv2.LINE_AA)

        out_msg = self._bridge.cv2_to_imgmsg(img, encoding=img_msg.encoding)
        out_msg.header = img_msg.header
        self._pub.publish(out_msg)


def main():
    rclpy.init()
    rclpy.spin(YoloOBBVisualizer())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
