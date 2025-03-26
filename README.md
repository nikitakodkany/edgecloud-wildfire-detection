# Edge-to-Cloud Wildfire Detection System

This project implements a real-time wildfire detection pipeline optimized for edge-to-cloud deployment using satellite imagery from the CEMS Wildfire Dataset. The system uses a U-Net model for semantic segmentation, trained and evaluated on preprocessed Sentinel-2 images.

With a focus on scalable inference, this solution incorporates distributed model training, ONNX Runtime deployment, performance profiling, and a Streamlit dashboard. It simulates edge-based image streaming to evaluate latency and memory usage — making it a strong demonstration of system-level thinking for AI deployment at scale.

## Performance Highlights

- Designed and deployed an end-to-end wildfire detection pipeline using Sentinel-2 satellite imagery and semantic segmentation with U-Net.

- Trained model using PyTorch with support for multi-GPU scaling and distributed data loading. Exported model to ONNX for framework-agnostic inference.

- Integrated ONNX Runtime and TensorRT for high-throughput, low-latency inference across multiple backends.

- Achieved a **3.2× speedup in inference throughput**:
  - From 28.01 ms/frame (CPU) to 8.74 ms/frame (GPU TensorRT)
  - Further optimized to 5.10 ms/frame using TensorRT FP16 at 128×128 resolution

- Benchmarked across multiple input resolutions (64×64 to 256×256) and hardware targets (CPU, CUDA, TensorRT) to analyze trade-offs in latency and memory usage.

- Reduced memory bottlenecks by **up to 42%** via optimized operator execution, resolution scaling, and frame-wise stream processing.

- Logged frame-by-frame inference latency and memory usage to structured `.txt` reports for system-level evaluation and reproducibility.

## Features

- **Dataset Integration:** Uses open-source Sentinel-2 satellite imagery from the CEMS Wildfire Dataset
- **Semantic Segmentation:** U-Net architecture trained to detect wildfire-affected regions at pixel level
- **Preprocessing Pipeline:** Converts raw GeoTIFF data into normalized, resized NumPy arrays with masks
- **Distributed Training:** Supports PyTorch-based multi-GPU training using DistributedDataParallel (DDP)
- **ONNX Model Export:** Converts PyTorch model to ONNX for inference acceleration
- **Stream Simulator:** Simulates image streaming from edge devices with real-time frame-by-frame inference
- **Performance Profiling:** Logs inference latency and memory usage per frame with summary metrics

## Pipeline Overview

![Pipeline Architecture](pipelinearch.png) 