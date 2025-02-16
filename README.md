# text-to-image-onnx
Optimizing a text-to-image diffusion model using ONNX and TensorRT

## Project Overview

This project explores optimizing Stable Diffusion for efficient text-to-image generation.
It focuses on converting the model from PyTorch to ONNX and further optimizing inference using TensorRT.

## Features

1. Uses Stable Diffusion Turbo (stabilityai/sd-turbo) for fast image generation.
2. Converts the UNet model to ONNX for deployment.
3. Currently optimizing inference with TensorRT for reduced latency.

## Ongoing Optimization: NVIDIA TensorRT for Faster Inference  

To improve inference speed and reduce memory usage, I am optimizing the ONNX-based **Stable Diffusion UNet model** using **NVIDIA TensorRT**.  

## Current Progress 
1. Converted the Stable Diffusion UNet model to **ONNX format**.  
2. Preparing to optimize ONNX model with **TensorRT** for efficient execution on NVIDIA GPUs.  

## Challenges & Debugging
1. **Precision Issues** – Testing FP16 vs. INT8 quantization for better performance.  
2. **Compatibility Checks** – Ensuring TensorRT supports all required ONNX operations.  
3. **Benchmarking** – Measuring inference speed improvement after conversion.  

## Next Steps  
1. Convert ONNX model to **TensorRT engine** (.trt file).  
2. Optimize execution with **batch size tuning & kernel selection**.  
3. Compare performance gains vs. standard PyTorch model.  
