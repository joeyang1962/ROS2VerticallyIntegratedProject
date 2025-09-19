#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class HandFollower(Node):
    """
    Hand-following controller that ONLY moves when something is close.
      - Subscribes: /scan  (LaserScan)
      - Publishes : /drive (AckermannDriveStamped)  <-- mux subscribes here
    Prints detailed info on EVERY scan so you can verify behavior.
    """

    def __init__(self):
        super().__init__('hand_follower')

        # ---- Parameters (override via --ros-args -p name:=value)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')          # <-- default to mux input
        self.declare_parameter('engage_distance', 0.60)          # start moving if closest < this [m]
        self.declare_parameter('target_distance', 0.35)          # maintain this distance [m]
        self.declare_parameter('kp_speed', 1.2)                  # m/s per meter error
        self.declare_parameter('kp_steer', 1.5)                  # rad per rad
        self.declare_parameter('max_speed', 0.30)                # m/s (keep small for first tests)
        self.declare_parameter('max_steering_angle', 0.34)       # rad (~19.5°)
        self.declare_parameter('front_angle_window_deg', 90.0)   # +/- around forward
        self.declare_parameter('allow_reverse', False)           # many VESC setups ignore reverse
        self.declare_parameter('angle_offset_deg', 0.0)          # if LiDAR 0° isn’t straight ahead

        # Resolve params
        self.scan_topic   = self.get_parameter('scan_topic').value
        self.drive_topic  = self.get_parameter('drive_topic').value
        self.engage_dist  = float(self.get_parameter('engage_distance').value)
        self.target       = float(self.get_parameter('target_distance').value)
        self.kp_v         = float(self.get_parameter('kp_speed').value)
        self.kp_w         = float(self.get_parameter('kp_steer').value)
        self.vmax         = float(self.get_parameter('max_speed').value)
        self.steer_max    = float(self.get_parameter('max_steering_angle').value)
        self.front_window = math.radians(float(self.get_parameter('front_angle_window_deg').value))
        self.allow_rev    = bool(self.get_parameter('allow_reverse').value)
        self.angle_offset = math.radians(float(self.get_parameter('angle_offset_deg').value))

        # QoS: LiDAR is typically BEST_EFFORT; keep publisher default RELIABLE
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, scan_qos)
        self.pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)

        self.get_logger().info(
            "HandFollower started:\n"
            f"  scan_topic:   {self.scan_topic}\n"
            f"  drive_topic:  {self.drive_topic}   (mux subscribes here)\n"
            f"  engage_dist:  {self.engage_dist:.2f} m | target: {self.target:.2f} m\n"
            f"  gains:        kp_speed={self.kp_v:.2f}, kp_steer={self.kp_w:.2f}\n"
            f"  limits:       vmax={self.vmax:.2f} m/s, steer_max={self.steer_max:.2f} rad\n"
            f"  front_window: ±{math.degrees(self.front_window):.1f}°, "
            f"angle_offset={math.degrees(self.angle_offset):.1f}°, "
            f"reverse={'on' if self.allow_rev else 'off'}"
        )

    def on_scan(self, msg: LaserScan):
        # Use device-provided ranges; accept very small but non-zero readings
        r_min = max(0.02, float(msg.range_min))
        r_max = float(msg.range_max)

        total = len(msg.ranges)
        valid_cnt = 0
        in_win_cnt = 0

        best_r: Optional[float] = None
        best_a: Optional[float] = None

        half = 0.5 * self.front_window
        a = msg.angle_min + self.angle_offset
        inc = msg.angle_increment

        # Search closest valid point within the front window
        for r in msg.ranges:
            valid = (r == r) and (r != float('inf')) and (r_min <= r <= r_max)
            if valid:
                valid_cnt += 1
                if -half <= a <= half:
                    in_win_cnt += 1
                    if best_r is None or r < best_r:
                        best_r, best_a = r, a
            a += inc

        # Print scan stats every callback
        self.get_logger().info(
            f"[SCAN] frame='{msg.header.frame_id}' "
            f"| N={total} valid={valid_cnt} in_front={in_win_cnt} "
            f"| range_min/max=({r_min:.3f},{r_max:.3f}) "
            f"| angle_min/max=({math.degrees(msg.angle_min):.1f}°, {math.degrees(msg.angle_max):.1f}°) "
            f"inc={math.degrees(inc):.2f}°"
        )

        # No valid detection in window -> STOP
        if best_r is None or best_a is None:
            self.get_logger().info("[LOGIC] No valid target in front window -> STOP")
            self.publish_cmd(0.0, 0.0)
            return

        self.get_logger().info(
            f"[TARGET] closest={best_r:.3f} m @ {math.degrees(best_a):.1f}° "
            f"(engage<{self.engage_dist:.2f} m?)"
        )

        # Not close enough to engage -> STOP
        if best_r > self.engage_dist:
            self.get_logger().info("[LOGIC] Not engaged (target farther than engage_distance) -> STOP")
            self.publish_cmd(0.0, 0.0)
            return

        # Control law: approach target distance; steer toward closest angle
        v_cmd = self.kp_v * (self.target - best_r)   # positive when too far
        if not self.allow_rev:
            v_cmd = max(0.0, v_cmd)                  # block reverse unless allowed
        v_cmd = clamp(v_cmd, -self.vmax, self.vmax)

        steer = clamp(self.kp_w * best_a, -self.steer_max, self.steer_max)

        # Extra safety if super close
        if best_r < (r_min + 0.05):
            v_cmd = min(v_cmd, 0.0)

        subs = self.pub.get_subscription_count()
        self.get_logger().info(
            f"[CMD] v={v_cmd:.2f} m/s  steer={math.degrees(steer):.1f}° "
            f"-> {self.drive_topic}  (subs={subs})"
        )
        self.publish_cmd(v_cmd, steer)

    def publish_cmd(self, speed: float, steering: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = HandFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = AckermannDriveStamped()
        stop.drive.speed = 0.0
        stop.drive.steering_angle = 0.0
        node.pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

