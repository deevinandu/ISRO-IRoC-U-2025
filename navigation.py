# navigation.py

import time
import math
import numpy as np
import config

class NavigationController:
    def __init__(self, mavlink_manager, vision_system, vio_ready_event, stop_event):
        self.mavlink = mavlink_manager
        self.vision = vision_system
        self.vio_ready_event = vio_ready_event
        self.stop_event = stop_event
        
        self.landings_completed = 0
        self.visited_spots_ned = []
        self.initial_z_offset = 0
        self.mission_target_z = 0
        self.current_node_index = 0
        self.max_nodes = 6

    def _check_failsafes(self):
        """
        Checks for low battery or RC link loss.
        Returns True if a failsafe is triggered, False otherwise.
        """

        voltage = self.mavlink.system_status.get('voltage_battery', 16800) 
        if voltage < 00000: 
            print(f"\n[FAILSAFE] Low battery detected! Voltage: {voltage / 1000:.2f}V. Aborting mission.")
            return True

        rc_channels = self.mavlink.rc_channels
        if rc_channels and \
           rc_channels.get('chan1_raw', 1) == 0 and \
           rc_channels.get('chan2_raw', 1) == 0 and \
           rc_channels.get('chan3_raw', 1) == 0 and \
           rc_channels.get('chan4_raw', 1) == 0:
            print(f"\n[FAILSAFE] RC link loss detected! Main channels are zero. Aborting mission.")
            return True

        return False

    def _wait_for_disarm(self, timeout=60):
        print("\n[NAV] Waiting for disarm...")
        start = time.time()
        while time.time() - start < timeout:
            if not self.mavlink.is_armed:
                print("[NAV] Drone disarmed.")
                return True
            time.sleep(1)
        print("[NAV] Timeout waiting for disarm.")
        return False

    def _wait_for_position_target(self, target_xy, target_z, tolerance=0.3, timeout=20):
        print(f"\n[NAV] Moving to Target -> N:{target_xy[0]:.2f}, E:{target_xy[1]:.2f}, D:{target_z:.2f}")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.stop_event.is_set() or self._check_failsafes(): return False 
            self.mavlink.send_position_target_local_ned(target_xy[0], target_xy[1], target_z)
            pos = self.vision.latest_vio_position_ned
            dist_xy = math.sqrt((pos[0] - target_xy[0])**2 + (pos[1] - target_xy[1])**2)
            dist_z = abs(pos[2] - target_z)
            if dist_xy < tolerance and dist_z < tolerance:
                print(f"[NAV] Target position reached.")
                return True
            time.sleep(1)
        print(f"[NAV] Timeout waiting for position.")
        return False

    def _wait_for_takeoff_completion(self, target_agl, timeout=30):
        print(f"[NAV] Monitoring takeoff to {target_agl}m AGL...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.stop_event.is_set(): return False
            current_z_ned = self.vision.latest_vio_position_ned[2]
            current_agl = -(current_z_ned - self.initial_z_offset)
            print(f"[NAV] Current AGL: {current_agl:.2f}m", end='\r')
            if current_agl >= target_agl * 0.4:
                print(f"\n[NAV] Takeoff complete. Reached {current_agl:.2f}m.")
                return True
            time.sleep(0.5)
        print("\n[NAV] Timeout during takeoff.")
        return False

    def _scan_for_landing_spot(self, duration_s):
        print(f"[NAV] Reached node. Braking and stabilizing for 2 seconds...")
        self.mavlink.send_velocity_command(0, 0, 0)
        time.sleep(2)
        print(f"[NAV] Stabilized. Now scanning for {duration_s} seconds...")
        scan_start_time = time.time()
        while time.time() - scan_start_time < duration_s:
            if self.stop_event.is_set() or self._check_failsafes(): return None 
            current_pos = self.vision.latest_vio_position_ned
            self.mavlink.send_position_target_local_ned(current_pos[0], current_pos[1], current_pos[2])
            spot_info = self.vision.latest_landing_spot_info
            if (new_spot_ned_xy := spot_info.get('newly_locked_spot_ned')) is not None:
                is_duplicate = any(np.linalg.norm(new_spot_ned_xy - np.array(visited)[:2]) < config.DUPLICATE_SPOT_DISTANCE_M for visited in self.visited_spots_ned)
                if not is_duplicate:
                    print(f"\n[NAV] New valid spot found at {new_spot_ned_xy}!")
                    return new_spot_ned_xy
            time.sleep(0.1)
        print("[NAV] Scan complete. No new spot found.")
        return None

    def _perform_landing_sequence(self, spot_ned):
        print("\n" + "="*50)
        print(f"[NAV] INITIATING LANDING SEQUENCE {self.landings_completed + 1}/{config.MAX_SAFE_SPOTS_TO_LAND}")
        print(f"      Target: {spot_ned[0]:.2f} N, {spot_ned[1]:.2f} E")
        print("="*50)
        print("[NAV] Spot locked! Stabilizing before approach...")
        self.mavlink.send_velocity_command(0, 0, 0)
        time.sleep(2)
        if not self._wait_for_position_target(spot_ned, self.mission_target_z):
            print("[NAV] Failed to navigate to landing spot. Resuming survey.")
            return False
        print("[NAV] Reached spot. Hovering for 5 seconds before landing...")
        time.sleep(5)
        print("[NAV] Landing...")
        self.mavlink.set_mode('LAND')
        if not self._wait_for_disarm():
            raise Exception("Failed to disarm after landing.")
        self.landings_completed += 1
        self.visited_spots_ned.append(spot_ned)
        if self.landings_completed >= config.MAX_SAFE_SPOTS_TO_LAND:
            print("[NAV] Maximum number of landings reached. Mission complete.")
            return True
        print("[NAV] Taking off to resume survey...")
        self.mavlink.set_mode('GUIDED')
        time.sleep(1); self.mavlink.arm(); time.sleep(1)
        self.mavlink.takeoff(config.TAKEOFF_ALTITUDE_M)
        if not self._wait_for_takeoff_completion(config.TAKEOFF_ALTITUDE_M):
            raise Exception("Failed to complete takeoff after landing.")
        print("\n" + "="*50)
        print("[NAV] RESUMING SURVEY")
        print("="*50)
        return False

    def execute_mission(self):
        print("[NAV] Navigation thread started. Waiting for Vision System...")
        self.vio_ready_event.wait()
        print("[NAV] Vision System is ready. Starting mission sequence.")

        try:
            self.mavlink.reset_ekf_origin_here()
            self.initial_z_offset = self.mavlink.local_position['z']
            print(f"[NAV] Ground level set. Initial Z_NED: {self.initial_z_offset:.2f}m")
            self.mavlink.set_mode('GUIDED'); time.sleep(2); self.mavlink.arm(); time.sleep(2)
            print("[NAV] Taking off...")
            self.mavlink.takeoff(config.TAKEOFF_ALTITUDE_M)
            if not self._wait_for_takeoff_completion(config.TAKEOFF_ALTITUDE_M):
                raise Exception("Failed to reach takeoff altitude.")
            self.mission_target_z = self.initial_z_offset - config.TAKEOFF_ALTITUDE_M
            print("[NAV] Hovering for 5 seconds after takeoff...")
            time.sleep(5)

            print("\n[NAV] --- STARTING NODE-BASED SURVEY ---")
            
            while self.current_node_index < self.max_nodes and \
                  self.landings_completed < config.MAX_SAFE_SPOTS_TO_LAND:
                
                if self._check_failsafes():
                    break

                self.current_node_index += 1
                
                print("\n" + "-"*20 + f" SURVEY NODE {self.current_node_index}/{self.max_nodes} " + "-"*20)
                node_x_position = float(self.current_node_index)
                
                if not self._wait_for_position_target([node_x_position, 0], self.mission_target_z):
                    print("[NAV] Failed to reach survey node. Aborting.")
                    break
                
                found_spot = self._scan_for_landing_spot(duration_s=10)
                
                if found_spot is not None:
                    last_node_position = [node_x_position, 0]
                    mission_is_over = self._perform_landing_sequence(found_spot)
                    if mission_is_over:
                        break
                    
                    print(f"[NAV] Returning to last survey node ({last_node_position[0]}N, {last_node_position[1]}E)...")
                    if not self._wait_for_position_target(last_node_position, self.mission_target_z):
                        print("[NAV] Failed to return to survey node. Aborting.")
                        break
            
            print("\n[NAV] Survey loop finished.")

        except Exception as e:
            print(f"\n[NAV] Mission error: {e}")
        finally:
            print("\n[NAV] --- MISSION END: RETURNING TO HOME AND LANDING ---")
            if self.mavlink.is_armed:
                 print(f"[NAV] Returning to Home (0,0) at {config.TAKEOFF_ALTITUDE_M}m AGL.")
                 if not self._wait_for_position_target([0, 0], self.mission_target_z):
                     print("[WARN] Could not confirm return to home position. Attempting to land anyway.")

                 print("[NAV] At home position. Hovering for 5 seconds...")
                 time.sleep(5)
                 self.mavlink.set_mode('LAND')
                 
            print("[NAV] Navigation thread finished.")
