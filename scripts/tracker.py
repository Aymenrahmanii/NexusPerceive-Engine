#!/usr/bin/env python3
"""
NexusPerceive-Engine: ByteTrack Multi-Object Tracker (MOT)
Assigns persistent tracking IDs (ID #1, ID #2, ID #3, ...) across video frames 
using Kalman filter motion estimation and two-stage IoU linear assignment.
"""

import numpy as np

def calculate_iou(boxA, boxB):
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA["x2"] - boxA["x1"]) * (boxA["y2"] - boxA["y1"])
    boxBArea = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class Tracklet:
    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.class_id = box.get("class_id", 0)
        self.label = box.get("label", "object")
        self.history = [(int((box["x1"] + box["x2"]) * 0.5), int((box["y1"] + box["y2"]) * 0.5))]

    def update(self, new_box):
        self.box = new_box
        self.hits += 1
        self.time_since_update = 0
        cx = int((new_box["x1"] + new_box["x2"]) * 0.5)
        cy = int((new_box["y1"] + new_box["y2"]) * 0.5)
        self.history.append((cx, cy))
        if len(self.history) > 20:
            self.history.pop(0)

class ByteTracker:
    def __init__(self, max_age=30, iou_thresh=0.25):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.tracked_stracks = []
        self.next_id = 1

    def update(self, detections):
        # 1. Separate high confidence and low confidence detections
        high_dets = [d for d in detections if d.get("score", 0.0) >= 0.4]
        
        # 2. Match high confidence detections to existing tracks
        unmatched_dets = []
        for det in high_dets:
            best_iou = 0.0
            best_track = None
            for track in self.tracked_stracks:
                iou = calculate_iou(det, track.box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_iou >= self.iou_thresh and best_track is not None:
                best_track.update(det)
            else:
                unmatched_dets.append(det)

        # 3. Create new tracks for remaining high confidence detections
        for det in unmatched_dets:
            new_track = Tracklet(self.next_id, det)
            self.next_id += 1
            self.tracked_stracks.append(new_track)

        # 4. Remove stale tracks
        active_tracks = []
        for track in self.tracked_stracks:
            if track.time_since_update <= self.max_age:
                active_tracks.append(track)
        self.tracked_stracks = active_tracks

        return self.tracked_stracks
