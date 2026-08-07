# NexusPerceive-Engine: Technical Architecture Specification

## Overview
NexusPerceive-Engine is a ultra-low-latency spatial perception engine designed for industrial real-time object detection (e.g. RT-DETR, YOLOv9) on NVIDIA GPUs (RTX 3090/4090, Jetson Orin AGX).

## System Latency Budget ($\le 4.2\text{ ms}$ / $\ge 235\text{ FPS}$)

| Stage | Mechanism | Target Latency | Bandwidth / Transfer |
|-------|-----------|----------------|----------------------|
| 1. Frame Ingestion | `cudaHostAllocMapped` Pinned RAM | $200\,\mu\text{s}$ | 6.22 MB (1080p BGR) |
| 2. Async H2D Transfer | DMA Stream `cudaMemcpyAsync` | $180\,\mu\text{s}$ | PCIe Gen4 x16 ($31.5\text{ GB/s}$) |
| 3. Fused CUDA Preproc | CUDA Grid `(40,40,1)` affine, NCHW FP16 | $280\,\mu\text{s}$ | 6.22 MB $\to$ 1.57 MB |
| 4. TensorRT Inference | `IExecutionContext::enqueueV3` FP16 | $2,800\,\mu\text{s}$ | Tensor Cores (RT-DETR FP16) |
| 5. Fused GPU NMS | Top-K + Shared Memory IoU suppression | $350\,\mu\text{s}$ | 8400 anchors $\to$ 100 boxes |
| 6. Async D2H Transfer | Filtered Bounding Boxes | $15\,\mu\text{s}$ | 2.4 KB ($100 \times 6 \times 4$ B) |
| 7. gRPC Transport | Binary Protobuf Stream | $240\,\mu\text{s}$ | Zero-Copy socket transport |
| **Total** | **End-to-End Pipeline** | **$4,065\,\mu\text{s}$ ($4.065\text{ ms}$)** | **$\sim 246\text{ FPS}$** |

## Triple-Buffering Stream Pipeline ($K=3$)

```
Stream 1 (Frame N):   [H2D] -> [Preproc] -> [TRT enqueueV3] -> [GPU NMS] -> [D2H]
Stream 2 (Frame N+1):          [H2D] -> [Preproc] -> [TRT enqueueV3] -> [GPU NMS] -> [D2H]
Stream 3 (Frame N+2):                   [H2D] -> [Preproc] -> [TRT enqueueV3] -> [GPU NMS] -> [D2H]
```

## Mathematical Foundations

### Symmetric Uniform Quantization (INT8 PTQ)
$$Q(X) = \text{clamp}\left(\left\lfloor \frac{X}{S} \right\rceil, -128, 127\right)$$

### Entropy Calibration via KL-Divergence
$$D_{\text{KL}}(P \parallel Q) = \sum_{i=1}^{N} P(i) \log \left( \frac{P(i)}{Q(i)} \right)$$

### CUDA Fused Preprocessor Affine Formula
$$x' = \frac{x - \text{pad}_w}{s}, \quad y' = \frac{y - \text{pad}_h}{s}, \quad \text{where } s = \min\left(\frac{W_{\text{out}}}{W_{\text{in}}}, \frac{H_{\text{out}}}{H_{\text{in}}}\right)$$
$$I_{\text{out}}(c, y, x) = \frac{\frac{I_{\text{in}}(x', y', c)}{255.0} - \mu_c}{\sigma_c}$$
