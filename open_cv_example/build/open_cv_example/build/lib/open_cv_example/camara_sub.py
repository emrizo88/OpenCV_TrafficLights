import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class WebcamViewer(Node):
    def __init__(self):
        super().__init__('webcam_viewer')
        self.subscription = self.create_subscription(Image,'/image_raw',self.image_callback,10)
        self.subscription  # Prevent unused variable warning
        self.cv_bridge = CvBridge()

    def image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow('Webcam', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    webcam_viewer = WebcamViewer()
    rclpy.spin(webcam_viewer)
    webcam_viewer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
