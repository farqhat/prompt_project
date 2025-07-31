from transformers import MarianMTModel, MarianTokenizer
from pathlib import Path

token_path = Path("hugging_face_token.txt")
tokenn = token_path.read_text().strip()


model_name = "Helsinki-NLP/opus-mt-ru-en"
model = MarianMTModel.from_pretrained(
    model_name,
    use_auth_token=tokenn
)
tokenizer = MarianTokenizer.from_pretrained(
    model_name,
    use_auth_token=tokenn
)

def translate_kz_to_en(text: str) -> str:
    tokens = tokenizer(text, return_tensors = "pt", padding = True, truncation = True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens = True)


