from huggingface_hub import login
login(token = "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")

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Provide a sample prompt
prompt = "What are your thoughts on the future of AI."

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate the response
output = model.generate(**inputs, max_new_tokens=100)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
