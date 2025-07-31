from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from ml import obtain_image, translate_kz_to_en
from pathlib import Path
import uuid

app = FastAPI()

# Create folder for saving images
output_dir = Path("generated_images")
output_dir.mkdir(exist_ok=True)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Stable Diffusion API!"}


@app.get("/generate")
def generate_image(
    prompt: str,
    translate_prompt: Optional[bool] = False,
    seed: Optional[int] = None,
    num_inference_steps: int = 5,
    guidance_scale: float = 7.5
):
    if translate_prompt:
        prompt = translate_kz_to_en(prompt)

    image, _ = obtain_image(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )

    filename = output_dir / f"image_{uuid.uuid4().hex[:8]}.png"
    image.save(filename)
    return FileResponse(filename)


class TranslationInput(BaseModel):
    text: str


@app.post("/translate")
def translate(input_data: TranslationInput):
    translated = translate_kz_to_en(input_data.text)
    return {"translated_text": translated} 