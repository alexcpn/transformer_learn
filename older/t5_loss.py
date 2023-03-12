from transformers import T5Tokenizer, T5ForConditionalGeneration

# https://huggingface.co/docs/transformers/model_doc/t5
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name,model_max_length=1024)
tokenizer.pad_token = tokenizer.eos_token
model = T5ForConditionalGeneration.from_pretrained(model_name)

# the following 2 hyperparameters are task-specific
max_source_length = 128
max_target_length = 20

# Suppose we have the following 2 training examples:
input_sequence_1  = "translate English to French: Welcome to NYC"
#output_sequence_1 = "Welcome to New York! Welcome to New York!"
output_sequence_1 = "Bienvenue à NYC"

# encode the inputs
encoding = tokenizer(
    input_sequence_1,
    padding=False,
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
)

input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
print(f"input_ids={input_ids}")
print(f"attention_mask={attention_mask}")
# encode the targets
target_encoding = tokenizer(
    output_sequence_1,
    padding=False,
    max_length=max_target_length,
    truncation=True,
    return_tensors="pt",
)
labels = target_encoding.input_ids
print(f"labels={labels}")
# replace padding token id's of the labels by -100 so it's ignored by the loss
labels[labels == tokenizer.pad_token_id] = -100
print(f"labels after={labels}")
# forward pass
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
print(f"Model Loss {loss.item()}")
#print(f"Model Loss {out.loss.item()}")
outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,max_new_tokens=max_target_length)
print(f"outputs[0] {outputs[0]}")
answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"Result {answer}")

"""
input_ids=tensor([[13959,  1566,    12,  2379,    10,  5242,    12, 13465,     1]])
attention_mask=tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
labels=tensor([[10520, 15098,     3,    85, 13465,     1]])
labels after=tensor([[10520, 15098,     3,    85, 13465,  -100]])
Model Loss 0.29697877168655396
outputs[0] tensor([    0, 10520, 15098,     3,    85,   368,  1060,     1])
Result Bienvenue à New York</s>
"""