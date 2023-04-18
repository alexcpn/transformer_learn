from transformers import GPT2LMHeadModel, GPT2Tokenizer

# https://huggingface.co/docs/transformers/model_doc/gpt2

tokenizer = GPT2Tokenizer.from_pretrained('gpt2',model_max_length=1024,padding_side='left')
tokenizer.pad_token = tokenizer.eos_token # == <|endoftext|> = 50256
model_name = './test-gpt2-2/gpt2-epoch-20-2023-04-15 16:31:57.913960' #Loss: 0.0506  # Model Loss
model = GPT2LMHeadModel.from_pretrained(model_name)

batch_size=5
input_text  = "Welcome to New York City"
target_text = "Welcome to New York City"

# encode the inputs
encoding = tokenizer(input_text,padding=True,max_length=batch_size,
                    truncation=True,return_tensors="pt",)
input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
# encode the targets
target_encoding = tokenizer(target_text,padding=True,max_length=batch_size,
                    truncation=True,return_tensors="pt",)
labels = target_encoding.input_ids
# replace padding token id's of the labels by -100 so it's ignored by the loss
labels[labels == tokenizer.pad_token_id] = -100  # in our case there is no padding
print(f"input_ids={input_ids}")
print(f"attention_mask={attention_mask}") # all ones
print(f"labels ={labels}")
# forward pass
outputs = model(input_ids=input_ids,labels=labels)
print(f"Model Loss {outputs.loss}")
# Test the model to check what it predicts next
# remove the last token off for input-id's as well as attention Mask
input_ids = input_ids[:,:-1] # input_text  = "Welcome to New York"
attention_mask = attention_mask[:,:-1]
print(f"input_ids={input_ids}")
outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,max_new_tokens=1)
answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"Result '{answer}'")
input_ids = input_ids[:,:-2] # "Welcome to"
test_output = model.generate(input_ids = input_ids,max_length=250,
                    num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
print(f"Result 2 '{test_answer}'")
