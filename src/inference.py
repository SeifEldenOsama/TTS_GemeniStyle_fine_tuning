import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from pathlib import Path

def generate_audio(text, description, model_path, output_path="output.wav", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Generate audio using the fine-tuned Parler-TTS model.
    """
    print(f"Loading model from {model_path}...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    print(f"Generating audio for: {text}")
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    description_input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=description_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    sf.write(output_path, audio_arr, model.config.sampling_rate)
    print(f"Audio saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    SAMPLE_TEXT = "Hello, this is a test of the fine-tuned Parler TTS model."
    SAMPLE_DESCRIPTION = "A female speaker with a clear and professional tone."
    MODEL_PATH = "parler-tts/parler-tts-mini-v1" # Replace with your fine-tuned path
    
    generate_audio(SAMPLE_TEXT, SAMPLE_DESCRIPTION, MODEL_PATH)
