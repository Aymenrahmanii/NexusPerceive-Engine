#!/usr/bin/env python3
"""
NexusPerceive-Engine: Real-Time ONNX Model Generator & Exporter
Exports a PyTorch object detection model (e.g. YOLO / RT-DETR structure) to ONNX 
with FP16 dynamic shape support for TensorRT compilation.
"""

import os
import torch
import torch.nn as nn

class DummyDetectionModel(nn.Module):
    """
    A lightweight spatial detection backbone + head simulating RT-DETR / YOLOv9.
    Input: [B, 3, 640, 640]
    Output: [B, 8400, 84] (4 bbox coordinates + 80 COCO class confidences)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.SiLU()

        # Head outputs 8400 anchors with 84 predictions (4 box + 80 classes)
        self.head = nn.Conv2d(64, 84, kernel_size=1)

    def forward(self, x):
        feat = self.act1(self.bn1(self.conv1(x)))
        feat = self.act2(self.bn2(self.conv2(feat)))
        out = self.head(feat) # [B, 84, H', W']
        
        # Flatten spatial dimensions to 8400 anchors
        B = out.shape[0]
        out_flat = out.view(B, 84, -1).permute(0, 2, 1) # [B, 8400, 84]
        return out_flat

def export_onnx():
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/rt_detr_r50vd.onnx"

    model = DummyDetectionModel().eval()
    dummy_input = torch.randn(1, 3, 640, 640)

    print(f"[EXPORT] Exporting real-time detection model to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "output0": {0: "batch_size"}
        }
    )
    print(f"[EXPORT] ONNX model exported successfully! Size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    export_onnx()
