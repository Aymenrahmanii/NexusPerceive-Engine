#!/usr/bin/env python3
"""
NexusPerceive-Engine: Real-Time ONNX Model Exporter for RT-DETR
Exports trained RT-DETR PCB Defect Detection model (or fallback dummy model) 
to ONNX format for TensorRT engine compilation and INT8 PTQ calibration.
"""

import os
import torch
import torch.nn as nn

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MY_COMPLIANT_INDUSTRIAL_RTDETR")

class DummyDetectionModel(nn.Module):
    """
    Fallback lightweight spatial detection backbone + head simulating RT-DETR.
    Input: [B, 3, 640, 640]
    Output: [B, 8400, 84]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.SiLU()

        self.head = nn.Conv2d(64, 84, kernel_size=1)

    def forward(self, x):
        feat = self.act1(self.bn1(self.conv1(x)))
        feat = self.act2(self.bn2(self.conv2(feat)))
        out = self.head(feat)
        B = out.shape[0]
        out_flat = out.view(B, 84, -1).permute(0, 2, 1)
        return out_flat

def export_onnx():
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/rt_detr_r50vd.onnx"

    if os.path.exists(MODEL_DIR):
        print(f"[EXPORT] Found trained RT-DETR model directory: {MODEL_DIR}")
        from transformers import RTDetrForObjectDetection
        model = RTDetrForObjectDetection.from_pretrained(MODEL_DIR)
        model.eval()
        dummy_input = torch.randn(1, 3, 640, 640)
        
        print(f"[EXPORT] Exporting industrial RT-DETR model to ONNX: {onnx_path}")
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_path,
            opset_version=17,
            input_names=["images"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "images": {0: "batch_size"},
                "logits": {0: "batch_size"},
                "pred_boxes": {0: "batch_size"}
            }
        )
    else:
        print(f"[EXPORT] Trained model directory not found. Exporting fallback dummy model...")
        model = DummyDetectionModel().eval()
        dummy_input = torch.randn(1, 3, 640, 640)
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

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"[EXPORT] ONNX model exported successfully! Path: {onnx_path} (Size: {size_mb:.2f} MB)")

if __name__ == "__main__":
    export_onnx()
