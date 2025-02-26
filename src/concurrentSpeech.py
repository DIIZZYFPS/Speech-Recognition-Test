import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
from concurrent.futures import ThreadPoolExecutor

# Load the model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# List of audio files to process
source = "src"
audio_files = ["src/harvard.wav", "src/jackhammer.wav"]

def process_audio_file(i, audio_file):
    start_time = time.time()
    result = pipe(audio_file)
    end_time = time.time()
    output_filename = f"output_{i+1}.txt"
    with open(f"test/{output_filename}", "w") as f:
        f.write(result["text"])
    print(f"Output saved to test/{output_filename}")
    print(f"Processing time for {audio_file}: {end_time - start_time:.2f} seconds")

# Process each audio file concurrently
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_audio_file, i, audio_file) for i, audio_file in enumerate(audio_files)]
    for future in futures:
        future.result()