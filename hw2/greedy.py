from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval().to(device)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

input_text_hedgehog = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
input_text_json = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

def greedy_decode(input_text, max_length=1000, eos_token_id=151645):
    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Initialize generated sequence with input_ids
    generated_ids = input_ids.clone()
    
    # Generate tokens one by one
    for _ in range(max_length - len(input_ids[0])):
        # Get model predictions
        with torch.no_grad():
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
        
        # Get the token with highest probability (greedy selection)
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Append the new token to generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # Stop if EOS token is generated
        if next_token.item() == eos_token_id:
            break
    
    # Move generated_ids back to CPU for decoding
    generated_ids = generated_ids.cpu()
    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# Example usage
generated_story = greedy_decode(input_text_hedgehog)
print("Generated story:")
print(generated_story)
print("\n" + "="*50 + "\n")

generated_json = greedy_decode(input_text_json)
print("Generated JSON:")
print(generated_json)
