import torch
from diffusers import StableDiffusionPipeline
from IPython.display import Image, display

# Load the Stable Diffusion model
model_id = "stabilityai/sd-turbo"  # Lightweight SD Model
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
pipe = pipe.to(device)

# Generate the image
prompt = "A futuristic city at sunset"
image = pipe(prompt).images[0]

# Save the image
image_path = "outputs/generated_image.png"
image.save(image_path)

# Display the image
display(Image(filename=image_path))

print(f"Image saved at {image_path}")
