# Use the generated NER keywords (named_entity_r.py) and NLTK based extraction of the sentences
# around the key-word (matching_keyword.py) to create a QA pair

import pandas as pd
import nltk

# Load the punkt tokenizer for sentence splitting
nltk.download('punkt')

# set the number of following sentences to include
num_following =5

# Example text
with open('./data/17921-0-cleaned.txt','r') as f:
    text =f.read()

# Split the text into sentences
sentences = nltk.sent_tokenize(text)
file_name = "./data/medical_ner.csv"
df= pd.read_csv(file_name,index_col=False)
mask = df['label'].isin(['ORG'])
df= df[mask]

results_df =pd.DataFrame();

# Find all sentences that contain the keyword and their neighboring sentences
for index, row in df.iterrows():
    keyword= row['text']
    if len(keyword) < 10: #skip short words
        continue
    print("keyword=",keyword)
    result =[]
    # For each keyword go through all sentences, is there a better way
    for i, sentence in enumerate(sentences):
        if keyword in sentence:
            end = min(len(sentences), i + num_following + 1)
            result.extend(sentences[i:end])
    row = {'question': keyword, 'answer':"".join(result)}
    results_df = results_df.append(row, ignore_index=True)

results_df.to_csv('./data/qa_dataset.csv',sep='|',index=False)