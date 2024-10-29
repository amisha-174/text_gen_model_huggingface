
from huggingface_hub import login

login(token = "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")

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import time

# # Load the model and tokenizer
# model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Provide input prompt
# prompt = "How will AI impact our future?"

# # Tokenize the input
# inputs = tokenizer(prompt, return_tensors="pt")

# # Start timing
# start_time = time.time()

# # Generate the response (Note: max_new_tokens is reduced for quicker response)
# output = model.generate(**inputs, max_new_tokens=50)

# # End timing
# end_time = time.time()

# # Decode and print the generated text and response time
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)
# print("Response Time:", end_time - start_time, "seconds")


from transformers import AutoModelForCausalLM, AutoTokenizer

# Model ID
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Define the conversation messages
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

# Tokenize the input
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

# Generate the output (running on CPU)
outputs = model.generate(inputs['input_ids'], max_new_tokens=20)

# Decode and print the output text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Load model and tokenizer
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# # Move model to GPU
# model.to("cpu")

# # Provide your input prompt
# prompt = "What is your favorite condiment?"

# # Tokenize input
# tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

# # Generate text
# generated_ids = model.generate(tokens['input_ids'], max_new_tokens=1000, do_sample=True)

# # Decode the generated tokens
# result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# print(result)

