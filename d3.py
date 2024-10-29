from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Start timing
start_time = time.time()

# Provide input text
input_text = "how AI will impact future?"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt")

# # Generate response
# output = model.generate(**inputs, max_new_tokens=50)

# Generate response
output = model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=True)


# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# End timing
end_time = time.time()

print(generated_text)
print("Response Time:", end_time - start_time, "seconds")
