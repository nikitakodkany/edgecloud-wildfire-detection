# Edge-to-Cloud Wildfire Detection System

This project implements a real-time wildfire detection pipeline optimized for edge-to-cloud deployment using satellite imagery from the CEMS Wildfire Dataset. The system uses a U-Net model for semantic segmentation, trained and evaluated on preprocessed Sentinel-2 images.

With a focus on scalable inference, this solution incorporates distributed model training, ONNX Runtime deployment, performance profiling, and a Streamlit dashboard. It simulates edge-based image streaming to evaluate latency and memory usage — making it a strong demonstration of system-level thinking for AI deployment at scale.

## Features

- **Dataset Integration:** Uses open-source Sentinel-2 satellite imagery from the CEMS Wildfire Dataset
- **Semantic Segmentation:** U-Net architecture trained to detect wildfire-affected regions at pixel level
- **Preprocessing Pipeline:** Converts raw GeoTIFF data into normalized, resized NumPy arrays with masks
- **Distributed Training:** Supports PyTorch-based multi-GPU training using DistributedDataParallel (DDP)
- **ONNX Model Export:** Converts PyTorch model to ONNX for inference acceleration
- **Stream Simulator:** Simulates image streaming from edge devices with real-time frame-by-frame inference
- **Performance Profiling:** Logs inference latency and memory usage per frame with summary metrics
