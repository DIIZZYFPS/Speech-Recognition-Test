import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import time

#torch.backends.cuda.enable_mem_efficient_sdp(False)
#torch.backends.cuda.enable_flash_sdp(False)

# Load the model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo" # This model is the most accurate, but also the slowest

#model_id = "openai/whisper-tiny.en" # This model is smaller and faster, but less accurate

#model_id = "openai/whisper-small.en" # This model is the sweet spot between speed and accuracy

model = WhisperForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
)
model.to(device)

processor = WhisperProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    chunk_length_s=30,
    stride_length_s=5,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    
)

try:
    # Try loading your specific model or a known Gemma IT model
    # Make sure you have accepted the license on Hugging Face Hub
    generator = pipeline(
        "text-generation",
        #model="google/gemma-3-1b-it-qat-int4-unquantized", # Your specific model if available
        model="google/gemma-3-4b-it", # Or another Gemma IT model
        torch_dtype=torch_dtype,
        device=device,
        
        

        # Add quantization config here if needed and not part of the model name
        # e.g., load_in_4bit=True (requires bitsandbytes)
    )
    # Get the tokenizer for chunking later
    gemma_tokenizer = generator.tokenizer
    # Get model's max length - Gemma 2 has 8192, Gemma 3 has more
    # Use a safe margin, e.g., 7000 for Gemma 2, or higher for Gemma 3
    MAX_TOKENS = 4096 # Adjust based on the specific Gemma model
    print(f"Using Gemma model: {generator.model.config._name_or_path}")

except Exception as e:
    print(f"Error loading Gemma model: {e}")
    print("Falling back to default summarizer (might not give note-like output).")
    # Fallback or handle error
    generator = pipeline("summarization", device=device, torch_dtype=torch_dtype) # Fallback
    gemma_tokenizer = generator.tokenizer # Use fallback tokenizer
    MAX_TOKENS = 1000 # Fallback max tokens
    IS_GEMMA = False # Flag to know if we are using Gemma
else:
    IS_GEMMA = True

# --- Function for Token-Based Chunking (Important!) ---
def split_text_by_tokens(text, tokenizer, max_tokens):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    current_chunk_start = 0
    while current_chunk_start < len(tokens):
        current_chunk_end = min(current_chunk_start + max_tokens, len(tokens))
        chunk_tokens = tokens[current_chunk_start:current_chunk_end]
        if chunk_tokens:
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        current_chunk_start = current_chunk_end
    return chunks

# --- Processing Loop ---
audio_files = ["harvard.wav", "jackhammer.wav", "I.mp3"]

for i, audio_file in enumerate(audio_files):
    # ... (ASR part: run pipe, save output_N.txt) ...
    start_time = time.time()
    print(f"\nProcessing ASR for {audio_file}...")
    result = pipe(audio_file) # Your existing ASR pipe call
    end_time = time.time()
    output_filename = f"output_{i+1}.txt"
    with open(f"../test/{output_filename}", "w", encoding='utf-8') as f:
        f.write(result["text"])
    print(f"Output saved to test/{output_filename}")
    print(f"Processing time for {audio_file}: {end_time - start_time:.2f} seconds")

    # --- Summarization Section ---
    start_time = time.time()
    print(f"Processing Summarization for {audio_file}...")
    full_text = result["text"]

    # Chunk the text using the CORRECT tokenizer and MAX_TOKENS for the loaded model
    chunks = split_text_by_tokens(full_text, gemma_tokenizer, MAX_TOKENS)
    print(f"Split text into {len(chunks)} chunks based on tokens for model.")

    summaries = []
    for chunk_idx, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {chunk_idx + 1}/{len(chunks)}...")
        if IS_GEMMA:
            # --- Define your Gemma Prompt ---
            # ----- Replace the old prompt definition with this one -----
            prompt = f"""Context: {chunk}

Instruction:
Analyze the context provided above. ONLY use information present in the context. Format your response using Markdown. Structure your response as follows:

Main Theme:
[Identify the single main topic or purpose of the text in one sentence]

Key Points:
- [Extract 8-12 important points, facts, or arguments as concise bullet points. Avoid repetition.]

Key Terms:
- [Extract important terms or phrases that are relevant to the context and define them. Bold key terms.]

Summary:
[Provide a brief 3-5 sentence overall summary paragraph based on the key points.]

Markdown Output:
Main Theme:""" # Guide the model to start outputting with "Main Theme:"
    # ---------------------------------------------------------------------------

            try:
                # Adjust generation parameters: Increase max_new_tokens slightly?
                # 300 might be tight for Theme + Points + Summary paragraph. Try 400-500?
                outputs = generator(prompt, max_new_tokens=700, do_sample=True, temperature=0.3) # Increased max_new_tokens slightly

                # Extract the generated text after the prompt's static part
                # Careful with slicing - need to account for the length of the static part
                static_prompt_part = """Context: [Placeholder for chunk text]

Instruction:
Analyze the context provided above. ONLY use information present in the context. Format your response using Markdown. Structure your response as follows:

Main Theme:
[Identify the single main topic or purpose of the text in one sentence]

Key Points:
- [Extract 8-12 important points, facts, or arguments as concise bullet points. Avoid repetition.]

Key Terms:
- [Extract important terms or phrases that are relevant to the context and define them. Bold key terms.]

Summary:
[Provide a brief 5-7 sentence overall summary paragraph based on the key points.]

Markdown Output:
Main Theme:"""
                # We need the length of the prompt *without* the chunk text to slice correctly
                # A simpler way might be to find the start of the generated content.
                # The model should ideally generate starting with the Main Theme content.
                # Let's assume the output starts reasonably after the prompt instruction.
                # A safer way: find the index of "Main Theme:" in the output if the model repeats it.
                full_output_text = outputs[0]['generated_text']
                generated_summary = ""
                # Try finding the start marker we added to the prompt in the full output
                start_marker = "Markdown Output:\nMain Theme:"
                start_index = full_output_text.find(start_marker)
                if start_index != -1:
                     # Get the text *after* the marker
                    generated_summary = full_output_text[start_index + len(start_marker):].strip()
                    # Prepend the marker title for clarity in the output file
                    generated_summary = "Main Theme:\n" + generated_summary
                else:
                     # Fallback if marker isn't found (model might behave unexpectedly)
                     # This might require more robust parsing based on model output patterns
                     print("Warning: Could not reliably slice prompt from output. Using basic slicing.")
                     generated_summary = full_output_text[len(prompt):].strip() # Less reliable


                summaries.append(generated_summary)

            except Exception as e:
                print(f"    Error generating with Gemma chunk {chunk_idx + 1}: {e}")
                summaries.append(f"[Error summarizing chunk {chunk_idx + 1}]")
        else:
             # Fallback to default summarizer pipeline if Gemma failed
             try:
                summary_output = generator(chunk, max_length=150, min_length=30)
                summaries.append(summary_output[0]["summary_text"])
             except Exception as e:
                print(f"    Error summarizing chunk {chunk_idx + 1} with fallback: {e}")
                summaries.append(f"[Error summarizing chunk {chunk_idx + 1}]")


    # Join summaries (adjust separator if needed, e.g., "\n\n" for bullet lists)
    summary_text = "\n".join(summaries)
    end_time = time.time()

    if len(chunks) > 1:
        # If we have multiple chunks, we need to synthesize them into a final summary
        print("Synthesizing final summary from chunk summaries...")
        reducer_prompt = f"""The following text consists of summaries from consecutive chunks of a longer document:

{summary_text}

Instruction:
Synthesize the information from the chunk summaries above into a single, cohesive final summary covering the main theme, key points, and key terms of the original document. Format the output using Markdown. Structure your response as follows:

### Overall Main Theme
[...]

### Overall Key Points
- [...]

### Overall Key Terms
- [...]

### Final Summary Paragraph
[...]

Final Synthesized Summary:
### Overall Main Theme"""
        
        try:
            final_summary_output = generator(reducer_prompt, max_new_tokens=1000, do_sample=True, temperature=0.3)
            final_summary_text = final_summary_output[0]['generated_text']


            # Extract the final summary text after the prompt's static part
            # Similar to the chunk summary extraction, but now we have a different prompt
            # We need the length of the prompt *without* the chunk text to slice correctly
            static_prompt_part = """The following text consists of summaries from consecutive chunks of a longer document:

[Placeholder for chunk summaries]

Instruction: 
Synthesize the information from the chunk summaries above into a single, cohesive final summary covering the main theme, key points, and key terms of the original document. Format the output using Markdown. Structure your response as follows:

### Overall Main Theme
[...]

### Overall Key Points
- [...]

### Overall Key Terms
- [...]

### Final Summary Paragraph
[...]

Final Synthesized Summary:
### Overall Main Theme"""
            final_generated_summary = ""
            # Find the start of the generated content
            start_marker = "Final Synthesized Summary:\n### Overall Main Theme"
            start_index = final_summary_text.find(start_marker)
            if start_index != -1:
                # Get the text *after* the marker
                final_generated_summary = final_summary_text[start_index + len(start_marker):].strip()
                # Prepend the marker title for clarity in the output file
                final_generated_summary = "### Overall Main Theme:\n" + final_generated_summary
            else:
                # Fallback if marker isn't found (model might behave unexpectedly)
                print("Warning: Could not reliably slice prompt from final output. Using basic slicing.")
                final_generated_summary = final_summary_text[len(reducer_prompt):].strip() # Less reliable


            summary_text = final_generated_summary
        except Exception as e:
            print(f"Error generating final summary: {e}")
            final_summary_text = "[Error generating final summary]"
            summary_text = final_summary_text

    # Save the summary
    summary_filename = f"summary_{i+1}.txt"
    with open(f"../test/{summary_filename}", "w", encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Summary saved to test/{summary_filename}")
    print(f"Summarization time for {audio_file}: {end_time - start_time:.2f} seconds")

print("\nProcessing finished.")