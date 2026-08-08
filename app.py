import os
import time
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

try:
    import spaces
except ImportError:
    spaces = None

def gpu_decorator(func):
    if spaces is not None:
        return spaces.GPU(func)
    return func

# Model directory path
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MY_COMPLIANT_INDUSTRIAL_RTDETR")

# Determine computing device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Color map for the 6 PCB industrial defect classes
DEFECT_COLOR_MAP = {
    "mouse_bite": "#EF4444",      # Crimson Red
    "spur": "#F59E0B",            # Amber / Orange
    "missing_hole": "#3B82F6",    # Royal Blue
    "short": "#EC4899",           # Hot Pink
    "open_circuit": "#8B5CF6",     # Deep Purple
    "spurious_copper": "#10B981"   # Emerald Green
}

# Global initialization of model and processor
print(f"[NEXUS_ENGINE] Initializing RT-DETR Defect Detection Model from: {MODEL_DIR}")
print(f"[NEXUS_ENGINE] Hardware target device: {DEVICE.upper()}")

try:
    processor = RTDetrImageProcessor.from_pretrained(MODEL_DIR)
    model = RTDetrForObjectDetection.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    MODEL_LOADED = True
    LOAD_ERROR_MSG = ""
    print("[NEXUS_ENGINE] Model and processor successfully loaded!")
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR_MSG = str(e)
    processor = None
    model = None
    print(f"[NEXUS_ENGINE] ERROR loading model: {e}")

@gpu_decorator
def run_nexus_perceive(image, conf_threshold=0.35):
    """
    Executes real-time industrial PCB defect detection using the custom-trained RT-DETR model.
    """
    if image is None:
        return None, "### ⚠️ Warning\nPlease upload a PCB or industrial inspection image to run perception."

    if not MODEL_LOADED:
        return None, f"### ❌ Model Initialization Error\nFailed to load model weights from `{MODEL_DIR}`.\n**Error:** `{LOAD_ERROR_MSG}`"

    start_time = time.perf_counter()
    
    # Ensure RGB PIL Image format
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image_rgb = image.convert("RGB")
    orig_w, orig_h = image_rgb.size

    # Preprocessing
    preprocess_start = time.perf_counter()
    inputs = processor(images=image_rgb, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

    # Model Inference
    infer_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    # Post-processing Object Detections
    post_start = time.perf_counter()
    target_sizes = torch.tensor([[orig_h, orig_w]], device=DEVICE)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=conf_threshold
    )[0]
    postproc_ms = (time.perf_counter() - post_start) * 1000.0

    total_ms = (time.perf_counter() - start_time) * 1000.0
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    # Draw detections on output image
    annotated_image = image_rgb.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    boxes = results["boxes"].cpu().numpy()

    detections_summary = {}
    box_count = len(scores)

    for score, label_id, box in zip(scores, labels, boxes):
        class_name = model.config.id2label.get(int(label_id), f"Class_{label_id}")
        color = DEFECT_COLOR_MAP.get(class_name, "#10B981")
        
        detections_summary[class_name] = detections_summary.get(class_name, 0) + 1
        
        xmin, ymin, xmax, ymax = box
        # Draw bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)
        
        # Format label text
        caption = f"{class_name} {score*100:.1f}%"
        
        # Calculate label background box
        text_bbox = draw.textbbox((xmin, ymin), caption)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        label_bg = [xmin, max(0, ymin - text_h - 6), xmin + text_w + 10, ymin]
        draw.rectangle(label_bg, fill=color)
        draw.text((xmin + 5, max(0, ymin - text_h - 4)), caption, fill="#FFFFFF")

    # Construct Telemetry & Defect Diagnostics Report
    defect_list_md = ""
    if box_count == 0:
        defect_list_md = "*No defects detected at confidence threshold >= " + f"{conf_threshold:.2f}*"
    else:
        for cname, count in detections_summary.items():
            color_hex = DEFECT_COLOR_MAP.get(cname, '#10B981')
            defect_list_md += f"- <span style='color:{color_hex}; font-weight:bold;'>■ {cname}</span>: `{count}` detected\n"

    telemetry_report = f"""
### ⚡ NexusPerceive Engine Telemetry & Defect Diagnostics
* **Hardware Target:** `{DEVICE.upper()}` ({'ZeroGPU / NVIDIA CUDA' if DEVICE == 'cuda' else 'CPU Host Execution'})
* **Total Detections:** `{box_count}` defects identified
* **End-to-End Latency:** `{total_ms:.2f} ms` (`{fps:.1f} FPS`)
* **Pipeline Breakdown:**
  - *Preprocessor (Image Rescale/Normalize):* `{preprocess_ms:.2f} ms`
  - *RT-DETR Neural Backbone & Head:* `{infer_ms:.2f} ms`
  - *Decoder Post-Processing & NMS:* `{postproc_ms:.2f} ms`

#### 🔍 Defect Class Summary:
{defect_list_md}
"""
    
    return annotated_image, telemetry_report

# Gradio Interface Setup
with gr.Blocks(title="⚡ NexusPerceive-Engine: RT-DETR Industrial Perception") as demo:
    gr.Markdown(
        """
        # ⚡ NexusPerceive-Engine: Industrial PCB Defect Perception Engine
        High-Throughput Spatial Object Detection powered by custom-trained **RT-DETR (ResNet Backbone)**.
        Detects **mouse_bite**, **spur**, **missing_hole**, **short**, **open_circuit**, and **spurious_copper** in real-time.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload PCB / Industrial Inspection Image")
            conf_slider = gr.Slider(
                minimum=0.05, maximum=1.0, value=0.35, step=0.05,
                label="Confidence Threshold", info="Filters detection outputs by confidence score"
            )
            btn = gr.Button("⚡ Run NexusPerceive Defect Detection", variant="primary")
        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="Visual Inspection & Defect Mapping")
            telemetry_out = gr.Markdown(value="*Upload an image and click Run to view real-time engine telemetry and defect diagnostics.*")
            
    btn.click(
        fn=run_nexus_perceive,
        inputs=[input_img, conf_slider],
        outputs=[output_img, telemetry_out]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

