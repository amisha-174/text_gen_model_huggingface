from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Load tokenizer and model
model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model on CPU
model = AutoModelForCausalLM.from_pretrained(model_name)

start_time = time.time()

# Provide input prompt
prompt = "What is the future of AI?"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate the response on CPU
output = model.generate(**inputs, max_new_tokens=50)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

end_time = time.time()

print(generated_text)
print("Response Time:", end_time - start_time, "seconds")

