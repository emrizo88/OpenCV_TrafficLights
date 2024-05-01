import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # Change 'image_raw' to the correct topic for your camera
            self.image_callback,
            10)
        self.subscription  # To prevent the topic from being garbage collected immediately
        self.cv_bridge = CvBridge()

        self.publisher = self.create_publisher(Image, '/processed_image', 10)

    def image_callback(self, msg):
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            processed_frame = self.detect_and_highlight_circles(frame)
            processed_msg = self.cv_bridge.cv2_to_imgmsg(processed_frame, 'bgr8')
            self.publisher.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def adjust_color_range(self, hsv_image, lower_color_range, upper_color_range):
        mask = cv2.inRange(hsv_image, lower_color_range, upper_color_range)
        return mask

    def detect_and_highlight_circles(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 11)

        color = (0, 0, 0)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                   param1=50, param2=30, minRadius=5, maxRadius=30)

        if circles is not None and len(circles[0]) > 0:
            filtered_circles = []
            for circle in circles[0]:
                x, y, r = circle
                circle_area = frame[int(y) - int(r):int(y) + int(r), int(x) - int(r):int(x) + int(r)]
                if np.any(circle_area):
                    hsv_circle = cv2.cvtColor(circle_area, cv2.COLOR_BGR2HSV)
                    mean_hsv = np.mean(hsv_circle, axis=(0, 1))
                    if (0 <= mean_hsv[0] < 30 or 160 <= mean_hsv[0] <= 180) or \
                       (35 <= mean_hsv[0] <= 55) or (60 <= mean_hsv[0] <= 85):
                        filtered_circles.append(circle)

            if len(filtered_circles) > 0:
                circle = filtered_circles[0]
                x, y, r = np.round(circle[0]), np.round(circle[1]), np.round(circle[2])

                circle_area = frame[int(y) - int(r):int(y) + int(r), int(x) - int(r):int(x) + int(r)]
                hsv_circle = cv2.cvtColor(circle_area, cv2.COLOR_BGR2HSV)
                mean_hsv = np.mean(hsv_circle, axis=(0, 1))

                if 0 <= mean_hsv[0] < 30 or 160 <= mean_hsv[0] <= 180:
                    color = (0, 0, 255)
                    color_name = "Rojo"
                elif 35 <= mean_hsv[0] <= 55:
                    color = (0, 255, 255)
                    color_name = "Amarillo"
                elif 60 <= mean_hsv[0] <= 85:
                    color = (0, 255, 0)
                    color_name = "Verde"

                cv2.circle(frame, (int(x), int(y)), int(r), color, 2)
                cv2.rectangle(frame, (int(x) - int(r), int(y) - int(r)), (int(x) + int(r), int(y) + int(r)), color, 2)

                if color in [(0, 0, 255), (0, 255, 255), (0, 255, 0)]:
                    cv2.putText(frame, f"Semaforo en {color_name}", (int(x) - int(r), int(y) - int(r) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

def main(args=None):
    rclpy.init(args=args)
    traffic_light_detector = TrafficLightDetector()
    rclpy.spin(traffic_light_detector)
    traffic_light_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
