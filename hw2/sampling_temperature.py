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

def temperature_sampling(input_text, max_length=1000, temperature=1.0, eos_token_id=151645):
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
            
            # Convert logits to probabilities using softmax
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            # Randomly sample one token based on probabilities
            next_token = torch.multinomial(next_token_probs, num_samples=1)
        
        # Add the new token to our sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # Stop if we generated the end token
        if next_token.item() == eos_token_id:
            break
    
    # Convert token IDs back to text
    generated_ids = generated_ids.cpu()
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Try generating a story with different temperatures
temperature = 1
print("Tempterature:", temperature)

print(f"Generating story")
generated_story = temperature_sampling(input_text_hedgehog, temperature=temperature)
print("Generated story:")
print(generated_story)
print("\n" + "="*50 + "\n")

# Try generating JSON with low temperature for more deterministic output
print("Generating JSON")
generated_json = temperature_sampling(input_text_json, temperature=temperature)
print("Generated JSON:")
print(generated_json) 