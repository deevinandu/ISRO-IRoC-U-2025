#!/usr/bin/env python3
import cv2
import os
import argparse


def create_video_from_frames(frames_folder, output_video, fps):
    frames = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    if not frames:
        print(f"No frames found in the folder: {frames_folder}")
        return

    first_frame = cv2.imread(frames[0])
    if first_frame is None:
        print(f"Failed to read the first frame: {frames[0]}")
        return
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error: VideoWriter could not be opened.")
        return

    print(f"Assembling video from frames in '{frames_folder}' at {fps} FPS...")
    for frame_file in frames:
        img = cv2.imread(frame_file)
        if img is None:
            print(f"Warning: Could not read frame: {frame_file}")
            continue
        out.write(img)

    out.release()
    print(f"Video saved as {output_video}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video from frames saved in a folder.')
    parser.add_argument('--frames_folder', type=str, default='landing_zone_output', help='Folder containing frame images.')
    parser.add_argument('--output', type=str, default='assembled_video.mp4', help='Name of the output video file.')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second for the output video.')
    args = parser.parse_args()

    create_video_from_frames(args.frames_folder, args.output, args.fps)
