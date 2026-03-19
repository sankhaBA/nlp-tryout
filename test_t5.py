from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Give it a raw navigation JSON string and see what it does WITHOUT fine-tuning
input_text = "translate navigation to instruction: action: turn direction: left distance: 5m landmark: escalator"

inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model output:", result)