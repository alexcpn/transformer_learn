import os
from datasets import Dataset
from transformers import AutoTokenizer

dir_path =  "/ssd/dataset_test/"


def load_code_files(dir_path):
    data = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".py"):
            filepath = os.path.join(dir_path, filename)
            with open(filepath, "r") as f:
                file_content = f.read()
                data.append({"text": file_content})
    return data

data = load_code_files(dir_path)
dataset = Dataset.from_list(data)

# Now you can proceed with tokenization
print("Number of files loaded:", len(dataset))
print("Raw text example:")
print(dataset[0]["text"])  # Print the first file's content

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Print the number of examples being processed
    print(f"Processing {len(examples['text'])} examples")
    
    # Use a longer max_length or remove truncation to see full content
    tokenized = tokenizer(
        examples["text"],
        truncation=False,  # Remove truncation temporarily
        padding=True,
        max_length=None,   # Remove max_length constraint
        return_tensors="pt",
    )
    
    return tokenized

# Tokenize with detailed logging
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"]
)

print("tokenized_dataset",tokenized_dataset)
# Inspect a specific row with more details
row = tokenized_dataset[0]
print("Number of tokens:", len(row["input_ids"]))
decoded_text = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
print("Decoded Text Length:", len(decoded_text))
print("Decoded Text:", decoded_text)
