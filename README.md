# Text-to-Image Model Optimization & ONNX Deployment

## Project Overview
This project focuses on optimizing a **text-to-image diffusion model** for efficient inference. Using **Stable Diffusion Turbo (stabilityai/sd-turbo)**, I converted the **UNet model to ONNX**, making it ready for deployment on different platforms, including mobile and GPU-accelerated environments.

## Features & Achievements
- **Stable Diffusion Turbo (stabilityai/sd-turbo)**: A lightweight model for **fast image generation**.
- **ONNX Model Conversion**: Transformed the **PyTorch-based UNet model** into ONNX format.
- **Efficient Model Execution**: Prepared the ONNX model for potential optimizations on **edge devices and GPUs**.
- **Challenges Addressed**:
  - Debugging ONNX conversion and ensuring **compatibility** with all required operations.
  - **Reducing memory overhead** while maintaining high-quality image generation.

## Implementation Steps
1. **Installed required dependencies**: PyTorch, TensorFlow, Diffusers, ONNX, and ONNX-TensorFlow.
2. **Loaded the Stable Diffusion model** using Hugging Face’s Diffusers library.
3. **Converted the UNet model to ONNX format**, enabling cross-platform deployment.
4. **Validated the ONNX model** to ensure consistency with the original PyTorch model.
5. **Deployment in Progress**: Actively working on optimizing and deploying the ONNX model.

## Work in Progress: TensorRT Optimization (Upcoming)
I plan to optimize the **ONNX-based UNet model** using **NVIDIA TensorRT** to:
- Reduce **inference latency** and improve real-time performance.
- Test **FP16 vs. INT8 quantization** for better memory efficiency.
- Compare inference speed between **ONNX runtime and TensorRT execution**.

## Next Steps
- Convert the ONNX model into a **TensorRT engine (.trt file)**.
- Tune execution parameters for optimized **batch size and kernel selection**.
- Benchmark **performance improvements** against standard PyTorch inference.

---

