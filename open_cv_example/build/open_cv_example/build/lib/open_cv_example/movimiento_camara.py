import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32

class CarController(Node):
    def __init__(self):
        super().__init__('car_controller')
        self.subscription = self.create_subscription(Int32, '/signal', self.signal_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
    def signal_callback(self, msg):
        signal_value = msg.data
        twist_msg = Twist()

        if signal_value == 1:
            twist_msg.linear.x = 0.1
        elif signal_value == 2:
            twist_msg.linear.x = 0.1 / 2
        elif signal_value == 3:
            twist_msg.linear.x = 0.0
        else:
            self.get_logger().warning('Received unknown signal value')

        self.publisher.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    car_controller = CarController()
    rclpy.spin(car_controller)
    car_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
