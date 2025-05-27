from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import math

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval().to(device)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

# Example prompts
input_text_hedgehog = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
input_text_json = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

def beam_search(input_text, max_length=1000, num_beams=4, length_penalty=1.0, eos_token_id=151645):
    # Convert input text to token IDs
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    batch_size = input_ids.shape[0]
    input_length = len(input_ids[0])
    vocab_size = model.config.vocab_size
    
    # Initialize beam state
    beam_scores = torch.zeros(batch_size, num_beams, device=device)
    beam_scores[:, 1:] = -1e9  # Make sure only first beam is active
    beam_scores = beam_scores.view(-1)  # Flatten to [batch_size * num_beams]
    
    # Initialize sequences with input_ids
    generated_ids = input_ids.repeat_interleave(num_beams, dim=0)
    
    # Track which beams are finished and their lengths
    beam_finished = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)
    beam_lengths = torch.ones(batch_size * num_beams, device=device) * input_length
    
    # Generate tokens one by one
    for step in range(max_length - input_length):
        with torch.no_grad():
            # Get model predictions
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Convert logits to log probabilities
            next_token_logprobs = F.log_softmax(next_token_logits, dim=-1)
            
            # Add current beam scores to get new scores
            next_scores = beam_scores.unsqueeze(-1) + next_token_logprobs  # [batch_size * num_beams, vocab_size]
            
            # Handle finished beams
            next_scores[beam_finished] = -1e9
            
            # Reshape scores to [batch_size, num_beams * vocab_size]
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            
            # Get top candidates
            next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1)
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Update beam state
            beam_scores = next_scores.view(-1)
            beam_indices = next_indices + torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            beam_indices = beam_indices.view(-1)
            
            # Update generated sequences
            generated_ids = torch.cat([
                generated_ids[beam_indices],
                next_tokens.view(-1, 1)
            ], dim=-1)
            
            # Update beam lengths and finished status
            beam_lengths = beam_lengths[beam_indices] + 1
            beam_finished = beam_finished[beam_indices] | (next_tokens.view(-1) == eos_token_id)
            
            # Apply length penalty to scores
            length_penalized_scores = beam_scores / (beam_lengths ** length_penalty)
            
            # Re-rank beams based on length-penalized scores
            if step > 0:  # Only re-rank after first token
                for b in range(batch_size):
                    start_idx = b * num_beams
                    end_idx = start_idx + num_beams
                    # Get scores and indices for this batch
                    batch_scores = length_penalized_scores[start_idx:end_idx]
                    batch_indices = torch.argsort(batch_scores, descending=True)
                    # Re-order beams
                    beam_indices = torch.arange(start_idx, end_idx, device=device)[batch_indices]
                    # Update sequences and scores
                    generated_ids[start_idx:end_idx] = generated_ids[beam_indices]
                    beam_scores[start_idx:end_idx] = beam_scores[beam_indices]
                    beam_lengths[start_idx:end_idx] = beam_lengths[beam_indices]
                    beam_finished[start_idx:end_idx] = beam_finished[beam_indices]
            
            # Stop if all beams are finished
            if beam_finished.all():
                break
    
    # Get best sequence for each batch based on length-penalized scores
    length_penalized_scores = beam_scores / (beam_lengths ** length_penalty)
    best_beam_indices = length_penalized_scores.view(batch_size, num_beams).argmax(dim=1)
    best_beam_indices = best_beam_indices + torch.arange(batch_size, device=device) * num_beams
    best_sequences = generated_ids[best_beam_indices]
    
    # Convert to text
    best_sequences = best_sequences.cpu()
    return tokenizer.decode(best_sequences[0], skip_special_tokens=True)

# Try generating with different beam sizes and length penalties
num_beams = 4
length_penalty = 1.0

print(f"Beam search with num_beams={num_beams}, length_penalty={length_penalty}")
print("Generating story:")
generated_story = beam_search(input_text_hedgehog, num_beams=num_beams, length_penalty=length_penalty)
print("Generated story:")
print(generated_story)
print("\n" + "="*50 + "\n")

print("Generating JSON:")
generated_json = beam_search(input_text_json, num_beams=num_beams, length_penalty=length_penalty)
print("Generated JSON:")
print(generated_json) 