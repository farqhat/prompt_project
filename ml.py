from transformers import MarianMTModel, MarianTokenizer
from diffusers import StableDiffusionPipeline
import torch

# Translation setup
model_name = "Helsinki-NLP/opus-mt-ru-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_kz_to_en(text: str) -> str:
    batch = tokenizer([text], return_tensors="pt", padding=True)
    generated_ids = model.generate(**batch)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Image generation setup
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
)
pipe = pipe.to("cpu")  # or "cuda" if you have a GPU

def obtain_image(prompt: str, num_inference_steps=30, seed=None, guidance_scale=7.5):
    generator = torch.manual_seed(seed) if seed is not None else None
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    return image