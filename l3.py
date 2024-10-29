

from huggingface_hub import login

login(token = "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")

from transformers import AutoModelForCausalLM, AutoTokenizer
import time 

# Load the tokenizer and model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

# Start timing
start_time = time.time()

# Provide input prompt
prompt = "How will AI impact future technology?"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate the response on CPU
output = model.generate(**inputs, max_new_tokens=50)

# End timing
end_time = time.time()

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
print("Response Time:", end_time - start_time, "seconds")