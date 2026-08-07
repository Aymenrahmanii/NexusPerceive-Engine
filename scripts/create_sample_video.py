#!/usr/bin/env python3
"""
NexusPerceive-Engine: Dynamic Sample Traffic Video Generator
Creates a realistic 1080p MP4 video with moving vehicles and pedestrians 
for live perception pipeline demonstration and benchmarking.
"""

import os
import cv2
import numpy as np

def create_sample_video(output_path="models/sample_traffic_video.mp4", duration_sec=10, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1920, 1080
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[VIDEO GEN] Generating 1080p synthetic traffic video ({total_frames} frames @ {fps} FPS)...")

    # Define initial positions & velocities for moving objects
    objects = [
        {"type": "person", "x": 200, "y": 600, "vx": 3, "vy": 0, "w": 80, "h": 180, "color": (200, 200, 250)},
        {"type": "car",    "x": 400, "y": 700, "vx": 8, "vy": 0, "w": 280, "h": 140, "color": (100, 255, 100)},
        {"type": "truck",  "x": 1200, "y": 750, "vx": -6, "vy": 0, "w": 380, "h": 180, "color": (255, 150, 100)},
        {"type": "bicycle", "x": 100, "y": 550, "vx": 4, "vy": 0, "w": 100, "h": 120, "color": (255, 255, 100)},
    ]

    for frame_idx in range(total_frames):
        # Create dark highway background with lane markings
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 35) # Dark asphalt

        # Draw road lanes
        cv2.rectangle(frame, (0, 500), (width, 950), (50, 50, 55), -1)
        # Dashed center line
        dash_offset = (frame_idx * 15) % 80
        for x in range(-dash_offset, width, 80):
            cv2.rectangle(frame, (x, 725), (x + 40, 735), (200, 200, 200), -1)

        # Update and draw moving objects
        for obj in objects:
            obj["x"] += obj["vx"]
            # Wrap around edges
            if obj["x"] > width + 100:
                obj["x"] = -obj["w"]
            elif obj["x"] < -obj["w"] - 100:
                obj["x"] = width

            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            # Draw object body
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), obj["color"], -1)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 2)
            cv2.putText(frame, obj["type"].upper(), (int(x + 10), int(y + 30)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        out.write(frame)

    out.release()
    print(f"[VIDEO GEN] Video generated successfully: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    create_sample_video()
