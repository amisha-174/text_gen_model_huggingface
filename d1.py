from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Load the model
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Load model and tokenizer, forcing them to use CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"  # Remove device_map as it's not needed for CPU
).to("cpu")  # Send the model to CPU

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Start timing
start_time = time.time()


# Prepare the prompt
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# Prepare the input using Qwen's specific chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize the input
model_inputs = tokenizer([text], return_tensors="pt").to("cpu")  # Send inputs to CPU

# Generate output from the model
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# Extract the generated tokens (output)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode the output tokens back into readable text
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# End timing
end_time = time.time()

# Print the response
print(response)
print("Response Time:", end_time - start_time, "seconds")

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import time

# model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# # Load the model and tokenizer
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"  # Use "cpu" if you don't have a GPU
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Simple prompt for text generation
# prompt = "What is the future of AI?"

# # Start timing
# start_time = time.time()

# # Tokenize the input prompt
# input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

# # Generate the output
# generated_ids = model.generate(
#     input_ids=input_ids,
#     max_new_tokens=512
# )

# # Decode the generated output
# response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# # End timing
# end_time = time.time()

# # Print response and response time
# print("Response:", response)
# print("Response Time:", end_time - start_time, "seconds")
