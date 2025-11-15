# frame_saver.py

import cv2
import os
import shutil
import config

class FrameSaver:
    def __init__(self):
        self.output_dir = config.JPG_OUTPUT_DIR
        self.frame_count = 0
        
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[FrameSaver] Saving processed frames to: {self.output_dir}")

    def save(self, frame):
        """Saves a single frame to the output directory."""
        filename = os.path.join(self.output_dir, f"frame_{self.frame_count:06d}.jpg")
        cv2.imwrite(filename, frame)
        self.frame_count += 1
