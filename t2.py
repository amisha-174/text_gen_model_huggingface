import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
model_name = "arcee-ai/SuperNova-Medius"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Provide input prompt
prompt = "How will technology evolve in the next decade?"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Start timing
start_time = time.time()

# Generate the response on CPU
output = model.generate(**inputs, max_new_tokens=50)

# End timing
end_time = time.time()

# Decode and print the generated text and response time
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
print("Response Time:", end_time - start_time, "seconds")
