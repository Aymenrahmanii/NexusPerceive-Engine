#!/usr/bin/env python3
"""
NexusPerceive-Engine: Protobuf / gRPC Message Protocol Structures
Implements binary serialization & deserialization for low-latency network streaming.
"""

import struct

class BoundingBoxPB:
    def __init__(self, x1=0.0, y1=0.0, x2=0.0, y2=0.0, confidence=0.0, class_id=0):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.confidence = float(confidence)
        self.class_id = int(class_id)

    def serialize(self):
        # 5 floats + 1 int32 = 24 bytes
        return struct.pack("fffffi", self.x1, self.y1, self.x2, self.y2, self.confidence, self.class_id)

    @classmethod
    def deserialize(cls, data):
        x1, y1, x2, y2, conf, cid = struct.unpack("fffffi", data)
        return cls(x1, y1, x2, y2, conf, cid)

class FrameRequestPB:
    def __init__(self, frame_id=0, timestamp_us=0, width=1920, height=1080, channels=3, image_bytes=b""):
        self.frame_id = int(frame_id)
        self.timestamp_us = int(timestamp_us)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.image_bytes = image_bytes

    def serialize(self):
        header = struct.pack("qqiiiI", self.frame_id, self.timestamp_us, self.width, self.height, self.channels, len(self.image_bytes))
        return header + self.image_bytes

    @classmethod
    def deserialize(cls, stream):
        header_size = struct.calcsize("qqiiiI")
        frame_id, ts, w, h, c, img_len = struct.unpack("qqiiiI", stream[:header_size])
        img_bytes = stream[header_size:header_size + img_len]
        return cls(frame_id, ts, w, h, c, img_bytes)

class DetectionResultPB:
    def __init__(self, frame_id=0, timestamp_us=0, end_to_end_latency_ms=0.0, boxes=None):
        self.frame_id = int(frame_id)
        self.timestamp_us = int(timestamp_us)
        self.end_to_end_latency_ms = float(end_to_end_latency_ms)
        self.boxes = boxes if boxes is not None else []

    def serialize(self):
        num_boxes = len(self.boxes)
        header = struct.pack("qqfI", self.frame_id, self.timestamp_us, self.end_to_end_latency_ms, num_boxes)
        box_bytes = b"".join([box.serialize() for box in self.boxes])
        return header + box_bytes

    @classmethod
    def deserialize(cls, data):
        header_size = struct.calcsize("qqfI")
        frame_id, ts, lat, num_boxes = struct.unpack("qqfI", data[:header_size])
        boxes = []
        box_size = 24
        offset = header_size
        for _ in range(num_boxes):
            b_data = data[offset:offset + box_size]
            boxes.append(BoundingBoxPB.deserialize(b_data))
            offset += box_size
        return cls(frame_id, ts, lat, boxes)
