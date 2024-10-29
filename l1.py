from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_id = "google/gemma-7b"

# Login to Hugging Face if necessary
# huggingface-cli login

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Move model to GPU if available
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Provide your input prompt
prompt = "What are the benefits of using AI in healthcare?"

# Tokenize the input
tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate text
generated_ids = model.generate(tokens['input_ids'], max_new_tokens=100, do_sample=True)

# Decode the generated tokens
result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(result)
