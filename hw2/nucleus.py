from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval().to(device)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

# Example prompts
input_text_hedgehog = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
input_text_json = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

def nucleus_sampling(input_text, max_length=1000, p=0.9, temperature=1.0, eos_token_id=151645):
    # Convert input text to token IDs
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    generated_ids = input_ids.clone()
    
    # Generate tokens one by one until max_length or EOS token
    for _ in range(max_length - len(input_ids[0])):
        with torch.no_grad():
            # Get model's predictions for next token
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature scaling to logits
            next_token_logits = next_token_logits / temperature
            
            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Create a mask for tokens to keep (those in the nucleus)
            nucleus_mask = cumulative_probs <= p
            
            # Add the first token that exceeds the threshold to the nucleus
            nucleus_mask[..., 1:] = nucleus_mask[..., :-1].clone()
            nucleus_mask[..., 0] = True
            
            # Get the indices of tokens in the nucleus
            nucleus_indices = sorted_indices[nucleus_mask]
            
            # Get the probabilities of tokens in the nucleus
            nucleus_probs = sorted_probs[nucleus_mask]
            
            # Normalize the nucleus probabilities
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            
            # Sample from the nucleus
            next_token_idx = torch.multinomial(nucleus_probs, num_samples=1)
            next_token = nucleus_indices[next_token_idx]
            
            # Reshape next_token to match generated_ids dimensions
            next_token = next_token.view(1, 1)
        
        # Add the new token to our sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # Stop if we generated the end token
        if next_token.item() == eos_token_id:
            break
    
    # Convert token IDs back to text
    generated_ids = generated_ids.cpu()
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

temperature = 0.5
p = 0.9

print("temp, p", temperature, p)

# Try generating a story with different nucleus sizes and temperatures
generated_story = nucleus_sampling(input_text_hedgehog, p=p, temperature=temperature)
print("Generated story:")
print(generated_story)
print("\n" + "="*50 + "\n")

# Try generating JSON with low temperature for more deterministic output
print("Generating JSON")
generated_json = nucleus_sampling(input_text_json, p=p, temperature=temperature)
print("Generated JSON:")
print(generated_json) 