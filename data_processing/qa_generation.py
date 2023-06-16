import pandas as pd
import nltk
from nltk import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer

nltk.download('punkt')  # Download the necessary resource for sentence tokenization
# Read the data file
file_path = './data/small_3.txt'
with open(file_path, 'r') as file:
    content = file.read()
print(f"content length {len(content)}")

# Tokenize into sentences using NLTK libraray
sentences = sent_tokenize(content)
num_rows = len(sentences)
print(f"{num_rows}\n")

# Create a pandas dataframe to hold the generated values
df = pd.DataFrame(index=range(num_rows+1), columns=['Question', 'Answer'])

# Use the SquaD trained model to generate questions for each sentence
#model_name = "valhalla/t5-base-qg-hl"
model_name = "valhalla/t5-base-e2e-qg"
# https://github.com/patil-suraj/question_generation
t5model = T5ForConditionalGeneration.from_pretrained(model_name)
t5tokenizer = T5Tokenizer.from_pretrained(model_name)

# Use the above model to generate questions
def generate_question(text):
    input_text = "generate question: " + text
    input_tokens = t5tokenizer.encode(input_text, return_tensors="pt")
    output_tokens = t5model.generate(input_tokens, max_new_tokens=500)
    question = t5tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return question


for i, text in enumerate(sentences):
    question = generate_question(text)
    # print(f"Q:{question}")
    # print(f"A:{text}")
    # Later NER to extract the NER from the text and select other texts with the term or also supply
    # surrounding sentences
    df.at[i, 'Question'] = question
    df.at[i, 'Answer'] = 'A:'+text
    if i // 100 == 0:
        print('Generated {i} QA pairs')

df.to_csv('generated_qa.csv')
