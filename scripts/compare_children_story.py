import modal
import os
from pathlib import Path

GPU_CONFIG = "H100:1"
VOLUME_NAME = "tts-dataset-storage"
MOUNT_PATH = Path("/data")
FINETUNED_MODEL_PATH = MOUNT_PATH / "parler-tts-finetuned-h100-ultra-optimized"
BASE_MODEL_NAME = "parler-tts/parler-tts-mini-v1"

REQUIREMENTS = [
    "torch==2.5.0",
    "torchaudio==2.5.0",
    "transformers==4.46.1",
    "parler-tts @ git+https://github.com/huggingface/parler-tts.git",
    "soundfile",
    "scipy"
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(*REQUIREMENTS)
)

app = modal.App("parler-tts-children-story-comparison", image=image)

@app.function(
    volumes={str(MOUNT_PATH): modal.Volume.from_name(VOLUME_NAME)},
    gpu=GPU_CONFIG,
    timeout=600
)
def run_comparison(prompt: str, description: str):
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def generate_audio(model_id, model_name_label):
        print(f"Loading {model_name_label} model...")
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

        print(f"Generating audio with {model_name_label}...")
        input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        
        filename = f"children_story_{model_name_label.lower().replace(' ', '_')}.wav"
        sf.write(filename, audio_arr, model.config.sampling_rate)
        
        with open(filename, "rb") as f:
            return f.read(), filename

    base_audio, base_file = generate_audio(BASE_MODEL_NAME, "Base")
    
    if not FINETUNED_MODEL_PATH.exists():
        return f"Error: Fine-tuned model not found at {FINETUNED_MODEL_PATH}. Did the training finish?", None

    ft_audio, ft_file = generate_audio(str(FINETUNED_MODEL_PATH), "Fine-tuned")
    
    return {
        "base": (base_audio, base_file),
        "finetuned": (ft_audio, ft_file)
    }

@app.local_entrypoint()
def main():
    # Children's story test case
    test_prompt = "Once upon a time, in a magical forest, there lived a tiny dragon who couldn't breathe fire. Instead, he breathed colorful bubbles that made everyone laugh!"
    test_description = "A warm, friendly female voice telling a story to children, with a gentle and expressive tone."
    
    print(f"Starting children's story comparison inference...")
    print(f"Prompt: {test_prompt}")
    print(f"Description: {test_description}")
    
    results = run_comparison.remote(test_prompt, test_description)
    
    if isinstance(results, str):
        print(results)
    else:
        for key, (audio_data, filename) in results.items():
            with open(filename, "wb") as f:
                f.write(audio_data)
            print(f"Saved {key} result to {filename}")
        print("\nComparison complete! The children's story audio files are ready.")
