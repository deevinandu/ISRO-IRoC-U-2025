# mavlink_manager.py
import time
from pymavlink import mavutil
import config

class MavlinkManager:
    def __init__(self, device, baud, source_system_id):
        print(f"Connecting to FC on {device}...")
        self.master = mavutil.mavlink_connection(device, baud=baud, source_system=source_system_id)
        
        self.attitude = {'roll': None, 'pitch': None, 'yaw': None}
        self.local_position = {'x': 0, 'y': 0, 'z': 0}
        self.is_armed = False
        self.mode = None
        self.distances_cm = {}
        self.rc_channels = {}
        self.system_status = {} 

    def wait_for_heartbeat(self):
        print("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        print(f"Connected to system {self.master.target_system}, component {self.master.target_component}")

    def update(self):
        """Processes all incoming MAVLink messages. Call this in the main loop."""
        while True:
            msg = self.master.recv_match(
                type=['AHRS2', 'DISTANCE_SENSOR', 'LOCAL_POSITION_NED', 'HEARTBEAT', 'SYS_STATUS', 'COMMAND_ACK', 'RC_CHANNELS'], 
                blocking=False)
            
            if not msg: break
            
            msg_type = msg.get_type()
            if msg_type == 'AHRS2':
                self.attitude['roll'] = msg.roll
                self.attitude['pitch'] = msg.pitch
                self.attitude['yaw'] = msg.yaw
            elif msg_type == 'DISTANCE_SENSOR':
                if msg.orientation == mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270:
                    self.distances_cm[msg.id] = msg.current_distance if msg.current_distance > msg.min_distance else None
            elif msg_type == 'LOCAL_POSITION_NED':
                self.local_position['x'] = msg.x
                self.local_position['y'] = msg.y
                self.local_position['z'] = msg.z
            elif msg_type == 'HEARTBEAT':
                self.is_armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                self.mode = mavutil.mode_string_v10(msg)
            elif msg_type == 'RC_CHANNELS':
                self.rc_channels = msg.to_dict()
            elif msg_type == 'SYS_STATUS':
                self.system_status = msg.to_dict()
    
    def get_lidar_altitude_m(self):
        valid_distances = [d for d in self.distances_cm.values() if d is not None]
        return (sum(valid_distances) / len(valid_distances)) / 100.0 if valid_distances else None

    def send_vision_position_estimate(self, position, attitude_rad, covariance, reset_counter=0):
        self.master.mav.vision_position_estimate_send(
            int(time.time() * 1e6), position[0], position[1], position[2],
            attitude_rad['roll'], attitude_rad['pitch'], attitude_rad['yaw'], covariance, reset_counter)

    def send_vision_speed_estimate(self, velocity_frd, covariance):
        self.master.mav.vision_speed_estimate_send(
            int(time.time() * 1e6), velocity_frd[0], velocity_frd[1], velocity_frd[2], covariance, 0)

    def set_mode(self, mode_name):
        mode_id = self.master.mode_mapping().get(mode_name)
        if mode_id is None: return False
        print(f"[CMT] Setting mode to {mode_name}...")
        self.master.set_mode(mode_id)
        return True

    def arm(self):
        print("[CMT] Arming motors...")
        self.master.arducopter_arm()
        self.master.motors_armed_wait()
        print("[STA] Drone armed.")

    def takeoff(self, altitude_m):
        print(f"[CMT] Sending takeoff command to {altitude_m}m...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, float('nan'), 0, 0, altitude_m)

    def send_velocity_command(self, vx, vy, vz):
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            3527, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)

    def send_position_target_local_ned(self, x, y, z):
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            3576, x, y, z, 0, 0, 0, 0, 0, 0, 0, 0)

    def reset_ekf_origin_here(self):
        print("[CMT] Setting EKF origin to current location...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_HOME, 0, 1, 0, 0, 0, 0, 0, 0)
        time.sleep(1)
