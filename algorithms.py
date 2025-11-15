# algorithms.py

import cv2
import depthai as dai
import numpy as np
from collections import deque
import time
import math
from datetime import timedelta
import threading
import queue

import config
from frame_saver import FrameSaver


class KeyFrame:
    def __init__(self, kf_id, pose, keypoints, descriptors):
        self.id = kf_id
        self.pose = pose
        self.kps = keypoints
        self.des = descriptors

class SlamBackend(threading.Thread):
    def __init__(self, keyframe_queue, slam_correction, correction_lock):
        super().__init__()
        self.daemon = True
        self.keyframe_queue = keyframe_queue
        self.slam_correction = slam_correction
        self.correction_lock = correction_lock
        self.map_keyframes = {}
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def run(self):
        print("[SLAM] Backend thread started.")
        while True:
            try:
                new_kf_data = self.keyframe_queue.get(timeout=1.0)
                
                gray_image = new_kf_data['image']
                keypoints = [cv2.KeyPoint(x=f.position.x, y=f.position.y, size=20) for f in new_kf_data['features']]
                keypoints, descriptors = self.orb.compute(gray_image, keypoints)

                if descriptors is None or len(keypoints) < 25:
                    continue

                new_keyframe = KeyFrame(new_kf_data['id'], new_kf_data['pose'], keypoints, descriptors)
                
                if len(self.map_keyframes) > 10:
                    best_match_id, max_inliers = -1, 0
                    
                    for kf_id, old_kf in self.map_keyframes.items():
                        if abs(kf_id - new_keyframe.id) < 15: continue
                        
                        matches = self.bf_matcher.knnMatch(new_keyframe.des, old_kf.des, k=2)
                        # Apply ratio test
                        good_matches = [m for m, n in matches if len(n) > 0 and m.distance < 0.75 * n.distance]
                        
                        if len(good_matches) > 20:
                            src_pts = np.float32([new_keyframe.kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([old_kf.kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            
                            E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=config.PIXELS_PER_RAD, pp=(config.IMG_WIDTH/2, config.IMG_HEIGHT/2), method=cv2.RANSAC, prob=0.999, threshold=1.0)
                            if mask is None: continue
                            inliers = np.count_nonzero(mask)
                            
                            if inliers > max_inliers:
                                max_inliers, best_match_id = inliers, kf_id

                    if max_inliers > 25:
                        print(f"\n--- [SLAM] LOOP CLOSURE! Current KF {new_keyframe.id} -> Map KF {best_match_id} ({max_inliers} inliers) ---")
                        old_pose = self.map_keyframes[best_match_id].pose['position']
                        current_pose = new_keyframe.pose['position']
                        correction = old_pose - current_pose
                        
                        with self.correction_lock:
                            self.slam_correction['position'] += correction[:2]
                            self.slam_correction['reset_counter'] += 1
                            print(f"--- [SLAM] Correction Vector: {correction[0]:.2f}N, {correction[1]:.2f}E. Reset Counter: {self.slam_correction['reset_counter']} ---")

                self.map_keyframes[new_keyframe.id] = new_keyframe
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SLAM-ERROR] An error occurred in SLAM backend: {e}")


class FeatureProcessor:
    def __init__(self):
        self.tracked_features = {}
        self.roi = {
            'x_min': config.IMG_WIDTH * (1 - config.ROI_SCALE) / 2, 'x_max': config.IMG_WIDTH * (1 + config.ROI_SCALE) / 2,
            'y_min': config.IMG_HEIGHT * (1 - config.ROI_SCALE) / 2, 'y_max': config.IMG_HEIGHT * (1 + config.ROI_SCALE) / 2
        }
    def process_features(self, features):
        new_feature_ids = {f.id for f in features}
        for f in features:
            if f.id not in self.tracked_features:
                self.tracked_features[f.id] = deque(maxlen=2)
            self.tracked_features[f.id].append(f.position)
        for fid in list(self.tracked_features.keys()):
            if fid not in new_feature_ids: del self.tracked_features[fid]
    def get_median_pixel_motion(self):
        motions = []
        for path in self.tracked_features.values():
            if len(path) == 2:
                curr_pos = path[1]
                if self.roi['x_min'] <= curr_pos.x <= self.roi['x_max'] and self.roi['y_min'] <= curr_pos.y <= self.roi['y_max']:
                    motions.append([curr_pos.x - path[0].x, curr_pos.y - path[0].y])
        return np.median(motions, axis=0) if motions else np.zeros(2)

class VisionOdometry:
    def __init__(self):
        self.position_local = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def update_position(self, local_velocity_xy, dt, correction_vector=np.zeros(2)):
        """Now accepts the SLAM correction vector."""
        self.position_local[0] += local_velocity_xy[0] * dt + correction_vector[0]
        self.position_local[1] += local_velocity_xy[1] * dt + correction_vector[1]

class SafeLandingProcessor:
    def __init__(self):
        self.state = "SEARCHING"
        self.locked_spot_pixels = None
        self.confirming_spot_pixels, self.confirmation_frames = None, 0
        self.confirmation_buffer = deque(maxlen=config.CONFIRMATION_FRAMES_REQUIRED)
        self.last_erosion_kernel_size = -1
        self.erosion_kernel = None
        self.cleaning_kernel = np.ones((config.CLEANING_KERNEL_SIZE, config.CLEANING_KERNEL_SIZE), np.uint8)
        self.edge_cleaning_kernel = np.ones((config.EDGE_CLEANING_KERNEL_SIZE, config.EDGE_CLEANING_KERNEL_SIZE), np.uint8)

    def _reset_to_searching(self):
        self.state = "SEARCHING"
        self.locked_spot_pixels = None
        self.confirming_spot_pixels, self.confirmation_frames = None, 0
        self.confirmation_buffer.clear()

    def _calculate_required_pixel_size(self, altitude_m):
        if altitude_m <= 0: return -1
        view_width_m = 2 * altitude_m * math.tan(np.deg2rad(config.HFOV_DEG / 2))
        return int(config.REQUIRED_LANDING_SIZE_M * (config.IMG_WIDTH / view_width_m))

    def _find_valid_spot(self, eroded_mask, original_edge_map):
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        best_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best_contour) > (config.IMG_WIDTH * config.IMG_HEIGHT * 0.90): return None
        M = cv2.moments(best_contour)
        if M["m00"] == 0: return None
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        boundary_x = int(config.IMG_WIDTH * config.UNSAFE_BOUNDARY_PCT)
        boundary_y = int(config.IMG_HEIGHT * config.UNSAFE_BOUNDARY_PCT)
        if not (boundary_x < cx < (config.IMG_WIDTH - boundary_x) and boundary_y < cy < (config.IMG_HEIGHT - boundary_y)): return None
        mask = np.zeros(original_edge_map.shape, dtype=np.uint8)
        cv2.drawContours(mask, [best_contour], -1, 255, -1)
        if cv2.mean(original_edge_map, mask=mask)[0] > config.MAX_NOISE_INTENSITY: return None
        return (cx, cy)

    def _pixel_to_local_ned(self, spot_px, drone_pos_ned, drone_yaw_rad, altitude_m):
        px_offset_x = spot_px[0] - config.IMG_WIDTH / 2
        px_offset_y = spot_px[1] - config.IMG_HEIGHT / 2
        angle_x = px_offset_x / config.PIXELS_PER_RAD
        angle_y = px_offset_y / config.PIXELS_PER_RAD
        disp_camera_x = np.tan(angle_x) * altitude_m
        disp_camera_y = np.tan(angle_y) * altitude_m
        disp_body = np.array([disp_camera_y, -disp_camera_x, 0.0])
        cy, sy = math.cos(drone_yaw_rad), math.sin(drone_yaw_rad)
        R_body_to_ned = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return drone_pos_ned[:2] + (R_body_to_ned @ disp_body)[:2]

    def process_frame(self, edge_map, altitude_m, drone_pos_ned, drone_yaw_rad):
        newly_locked_spot_ned = None
        cleaned_edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_OPEN, self.edge_cleaning_kernel)
        output_frame = cv2.cvtColor(cleaned_edge_map, cv2.COLOR_GRAY2BGR)

        if altitude_m is None or altitude_m < config.MIN_DETECTION_ALTITUDE_M:
            self._reset_to_searching()
            cv2.putText(output_frame, "TOO LOW TO SEARCH", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return {'processed_frame': output_frame, 'newly_locked_spot_ned': None}

        required_px = self._calculate_required_pixel_size(altitude_m)
        is_worth_searching = (required_px > 4 and required_px < min(config.IMG_WIDTH, config.IMG_HEIGHT))
        
        previous_state = self.state
        current_spot_found_px = None
        if is_worth_searching:
            _, threshold_edges = cv2.threshold(cleaned_edge_map, config.EDGE_THRESHOLD, 255, cv2.THRESH_TOZERO)
            inverted_mask = cv2.bitwise_not(threshold_edges)
            cleaned_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_OPEN, self.cleaning_kernel)
            if self.last_erosion_kernel_size != required_px:
                self.erosion_kernel = np.ones((required_px, required_px), np.uint8)
                self.last_erosion_kernel_size = required_px
            eroded_mask = cv2.erode(cleaned_mask, self.erosion_kernel)
            current_spot_found_px = self._find_valid_spot(eroded_mask, edge_map)

        if self.state == "SEARCHING" and current_spot_found_px:
            self.state = "CONFIRMING"
            self.confirming_spot_pixels = current_spot_found_px
            self.confirmation_frames = 1
            self.confirmation_buffer.append(current_spot_found_px)
        elif self.state == "CONFIRMING":
            if current_spot_found_px and np.linalg.norm(np.array(current_spot_found_px) - np.array(self.confirming_spot_pixels)) < (required_px * 0.5):
                self.confirmation_frames += 1
                self.confirmation_buffer.append(current_spot_found_px)
                self.confirming_spot_pixels = current_spot_found_px
                if self.confirmation_frames >= config.CONFIRMATION_FRAMES_REQUIRED:
                    self.state = "LOCKED"
                    self.locked_spot_pixels = (int(np.median([p[0] for p in self.confirmation_buffer])), int(np.median([p[1] for p in self.confirmation_buffer])))
            else: self._reset_to_searching()
        elif self.state == "LOCKED":
            if not current_spot_found_px or np.linalg.norm(np.array(current_spot_found_px) - np.array(self.locked_spot_pixels)) > required_px:
                self._reset_to_searching()
        
        if not is_worth_searching: self._reset_to_searching()

        if self.state == "LOCKED" and previous_state != "LOCKED":
            locked_ned = self._pixel_to_local_ned(self.locked_spot_pixels, drone_pos_ned, drone_yaw_rad, altitude_m)
            if locked_ned is not None:
                if np.linalg.norm(locked_ned) < config.HOME_EXCLUSION_RADIUS_M:
                    print(f"[ALGO] Spot ignored: Too close to home (Dist: {np.linalg.norm(locked_ned):.2f}m).")
                    self._reset_to_searching()
                else:
                    newly_locked_spot_ned = locked_ned
        
        status_color, confirm_color, lock_color, center_color = (0, 255, 255), (0, 255, 255), (0, 255, 0), (0, 0, 255)
        lidar_text = f"Lidar: {altitude_m:.2f}m" if altitude_m is not None else "Lidar: N/A"
        cv2.putText(output_frame, lidar_text, (config.IMG_WIDTH - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        if self.state == "CONFIRMING":
            progress = self.confirmation_frames / config.CONFIRMATION_FRAMES_REQUIRED
            cv2.putText(output_frame, f"CONFIRMING ({progress:.0%})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, confirm_color, 2)
            if self.confirming_spot_pixels and required_px > 0:
                cx, cy = self.confirming_spot_pixels; half_size = required_px // 2
                cv2.rectangle(output_frame, (cx - half_size, cy - half_size), (cx + half_size, cy + half_size), confirm_color, 2)
        elif self.state == "LOCKED":
            cv2.putText(output_frame, "LOCKED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lock_color, 2)
            if self.locked_spot_pixels and required_px > 0:
                cx, cy = self.locked_spot_pixels; half_size = required_px // 2
                cv2.rectangle(output_frame, (cx - half_size, cy - half_size), (cx + half_size, cy + half_size), lock_color, 3)
                cv2.circle(output_frame, (cx, cy), 5, center_color, -1)
        else:
            cv2.putText(output_frame, "SEARCHING", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return {'processed_frame': output_frame, 'newly_locked_spot_ned': newly_locked_spot_ned}

# vision processing 

class VisionSystem:
    def __init__(self, mavlink_manager, ready_event, stop_event):
        self.mavlink = mavlink_manager
        self.ready_event = ready_event
        self.stop_event = stop_event
        self.odometry = VisionOdometry()
        self.feature_proc = FeatureProcessor()
        self.landing_proc = SafeLandingProcessor()
        self.frame_saver = FrameSaver() if config.SAVE_FRAMES else None
        self.keyframe_queue = queue.Queue(maxsize=5)
        self.slam_correction = {'position': np.zeros(2), 'reset_counter': 0}
        self.correction_lock = threading.Lock()
        self.slam_backend = SlamBackend(self.keyframe_queue, self.slam_correction, self.correction_lock)
        self.keyframe_id_counter = 0
        self.last_keyframe_pos = np.zeros(3)
        self.initial_keyframe_created = False
        self.last_frame_time = time.monotonic()
        self.is_initialized = False
        self.initial_yaw_rad = None
        self.latest_vio_position_ned = np.zeros(3)
        self.latest_vio_yaw_rad = 0.0
        self.latest_landing_spot_info = {}

    def create_pipeline(self):
        print("[ALGO] Creating VIO+SLAM+Landing pipeline...")
        pipeline = dai.Pipeline()
        mono = pipeline.create(dai.node.MonoCamera); imu = pipeline.create(dai.node.IMU)
        tracker = pipeline.create(dai.node.FeatureTracker); sync = pipeline.create(dai.node.Sync)
        edgeDetector = pipeline.create(dai.node.EdgeDetector)
        xout_sync = pipeline.create(dai.node.XLinkOut); xout_sync.setStreamName("vio_synced")
        xout_edge = pipeline.create(dai.node.XLinkOut); xout_edge.setStreamName("edge")
        
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        mono.setCamera("left"); mono.setFps(config.VIDEO_FPS)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, config.IMU_FREQUENCY)
        imu.setBatchReportThreshold(1)
        sync.setSyncThreshold(timedelta(milliseconds=10))
        edgeDetector.initialConfig.setSobelFilterKernels([[1, 0, -1], [2, 0, -2], [1, 0, -1]], [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        
        tracker.setHardwareResources(1, 1)      
        tracker.setWaitForConfigInput(False)
        tracker.passthroughInputImage.link(sync.inputs["image"])
        sync.inputs['image'].setBlocking(False)
        sync.inputs['image'].setQueueSize(2)
        
        mono.out.link(tracker.inputImage)
        mono.out.link(edgeDetector.inputImage)
        edgeDetector.outputImage.link(xout_edge.input)
        tracker.outputFeatures.link(sync.inputs["features"])
        imu.out.link(sync.inputs["imu"])
        sync.out.link(xout_sync.input)
        
        print("[ALGO] Pipeline created.")
        return pipeline

    def should_create_keyframe(self, current_position, altitude_m):
        if altitude_m is None or altitude_m < config.KEYFRAME_ALTITUDE_THRESHOLD:
            return False
        if not self.initial_keyframe_created:
            self.initial_keyframe_created = True
            self.last_keyframe_pos = current_position.copy()
            return True
        dist = np.linalg.norm(current_position - self.last_keyframe_pos)
        if dist > config.KEYFRAME_DISTANCE_THRESHOLD:
            self.last_keyframe_pos = current_position.copy()
            return True
        return False

    def translational_velocity(self, total_px_motion, gyro_rad_s, altitude_m, dt):
        if altitude_m <= 0.1 or dt <= 0: return np.zeros(2)
        px_motion_rot_x = gyro_rad_s.x * config.PIXELS_PER_RAD * dt
        px_motion_rot_y = gyro_rad_s.y * config.PIXELS_PER_RAD * dt
        px_motion_trans = np.array([total_px_motion[0] - px_motion_rot_x, total_px_motion[1] - px_motion_rot_y])
        angle_rad_x = px_motion_trans[0] / config.PIXELS_PER_RAD
        angle_rad_y = px_motion_trans[1] / config.PIXELS_PER_RAD
        disp_x = np.tan(angle_rad_x) * altitude_m
        disp_y = np.tan(angle_rad_y) * altitude_m
        return np.array([disp_y / dt, -disp_x / dt])
    
    def run(self):
        self.slam_backend.start()
        pipeline = self.create_pipeline()
        with dai.Device(pipeline) as device:
            sync_q = device.getOutputQueue("vio_synced", 8, False)
            edge_q = device.getOutputQueue("edge", 8, False)
            print("[ALGO] Vision System thread started.")
            
            while not self.stop_event.is_set():
                try:
                    loop_start_time = time.monotonic()
                    dt = loop_start_time - self.last_frame_time
                    self.last_frame_time = loop_start_time
                    
                    if not self.is_initialized and self.mavlink.attitude['yaw'] is not None:
                        self.initial_yaw_rad = self.mavlink.attitude['yaw'] + config.HEADING_OFFSET_RAD
                        self.is_initialized = True
                        self.ready_event.set()
                        print(f"\n[ALGO] Vision local frame initialized. Corrected heading: {math.degrees(self.initial_yaw_rad):.2f}°")
                    if not self.is_initialized:
                        time.sleep(0.1); continue
                    
                    ground_dist_m = self.mavlink.get_lidar_altitude_m()
                    current_attitude_rad = self.mavlink.attitude
                    
                    body_vel_xy = np.zeros(2)
                    synced_msg = sync_q.tryGet()
                    if synced_msg and ground_dist_m is not None:
                        correction_to_apply = np.zeros(2)
                        with self.correction_lock:
                            if np.linalg.norm(self.slam_correction['position']) > 1e-4:
                                correction_to_apply = self.slam_correction['position'] * 0.1
                                self.slam_correction['position'] *= 0.9

                        feature_data = synced_msg["features"].trackedFeatures
                        imu_data = synced_msg["imu"].packets[-1].gyroscope
                        
                        self.feature_proc.process_features(feature_data)
                        total_px_motion = self.feature_proc.get_median_pixel_motion()
                        body_vel_xy = self.translational_velocity(total_px_motion, imu_data, ground_dist_m, dt)
                        if np.linalg.norm(body_vel_xy) < config.VELOCITY_DEADZONE: body_vel_xy[:] = 0.0

                        local_yaw_rad = current_attitude_rad['yaw'] - self.initial_yaw_rad
                        cy, sy = math.cos(local_yaw_rad), math.sin(local_yaw_rad)
                        local_vel_xy = np.array([body_vel_xy[0] * cy - body_vel_xy[1] * sy, 
                                                 body_vel_xy[0] * sy + body_vel_xy[1] * cy])
                        
                        self.odometry.update_position(local_vel_xy, dt, correction_to_apply)
                    
                        self.latest_vio_position_ned = np.array([self.odometry.position_local[0], self.odometry.position_local[1], -ground_dist_m])
                        self.latest_vio_yaw_rad = (local_yaw_rad + math.pi) % (2 * math.pi) - math.pi

                        if self.should_create_keyframe(self.latest_vio_position_ned, ground_dist_m):
                            if not self.keyframe_queue.full():
                                self.keyframe_id_counter += 1
                                print(f"\n--- [ALGO] Creating KeyFrame #{self.keyframe_id_counter} at Pos: {self.latest_vio_position_ned[0]:.2f}N, {self.latest_vio_position_ned[1]:.2f}E ---")
                                kf_data = {
                                    'id': self.keyframe_id_counter, 
                                    'pose': {'position': self.latest_vio_position_ned, 'attitude': self.latest_vio_yaw_rad},
                                    'image': synced_msg["image"].getCvFrame(), 
                                    'features': synced_msg["features"].trackedFeatures
                                }
                                self.keyframe_queue.put(kf_data)

                    edge_msg = edge_q.tryGet()
                    if edge_msg:
                        self.latest_landing_spot_info = self.landing_proc.process_frame(
                            edge_msg.getCvFrame(), ground_dist_m, self.latest_vio_position_ned, self.latest_vio_yaw_rad)
                        if self.frame_saver:
                            self.frame_saver.save(self.latest_landing_spot_info['processed_frame'])

                    reset_counter_val = 0
                    with self.correction_lock:
                        reset_counter_val = self.slam_correction['reset_counter']
                    
                    last_sent_velocity_frd = np.array([body_vel_xy[0], body_vel_xy[1], math.nan])
                    local_attitude = {'roll': current_attitude_rad['roll'], 'pitch': current_attitude_rad['pitch'], 'yaw': self.latest_vio_yaw_rad}

                    self.mavlink.send_vision_speed_estimate(last_sent_velocity_frd, [0.0]*9)
                    self.mavlink.send_vision_position_estimate(self.latest_vio_position_ned, local_attitude, [0.0]*21, reset_counter_val)
                    
                    loop_duration = time.monotonic() - loop_start_time
                    sleep_time = (1.0 / config.VIO_RATE_HZ) - loop_duration
                    if sleep_time > 0: time.sleep(sleep_time)

                except Exception as e:
                    print(f"[ALGO-ERROR] An error occurred in VisionSystem run loop: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1) 
                    
        print("[ALGO] Vision System thread finished.")
