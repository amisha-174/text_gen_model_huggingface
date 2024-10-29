from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Start timing
start_time = time.time()

# Prepare input text
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
outputs = model.generate(**inputs, max_length=50, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# End timing
end_time = time.time()


print("Generated Text:", generated_text)
print("Response Time:", end_time - start_time, "seconds")
