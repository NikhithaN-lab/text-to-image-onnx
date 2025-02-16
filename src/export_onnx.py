import torch
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion model
model_id = "stabilityai/sd-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
pipe = pipe.to(device)

# Extract UNet model
unet = pipe.unet.to(torch.float32)

# Prepare ONNX Inputs
latent_input = torch.randn(1, 4, 64, 64).to(device, dtype=torch.float32)
timestep = torch.tensor([0], dtype=torch.int64).to(device)

# Generate text embeddings
prompt = "A futuristic city at sunset"
text_inputs = pipe.tokenizer(
    prompt,
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
text_input_ids = text_inputs.input_ids.to(device)
encoder_hidden_states = pipe.text_encoder(text_input_ids)[0].to(torch.float32)

# Export UNet to ONNX
onnx_model_path = "models/unet_model.onnx"
torch.onnx.export(
    unet,
    (latent_input, timestep, encoder_hidden_states),
    onnx_model_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["latent_input", "timestep", "encoder_hidden_states"],
    output_names=["output"],
    dynamic_axes={
        "latent_input": {0: "batch_size"},
        "timestep": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

print(f" Model exported to {onnx_model_path}")
