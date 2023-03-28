# Use the generated NER keywords (named_entity_r.py) and NLTK based extraction of the sentences
# around the key-word (matching_keyword.py) to create a QA pair

import pandas as pd
import nltk
from  utils import get_batch_for_qa
from transformers import  T5Tokenizer

def create_qa_dataframe(in_ner_file,dataframe_csv):
    # Load the punkt tokenizer for sentence splitting
    nltk.download('punkt')
    
    # set the number of following sentences to include
    num_following =5

    # Example text
    with open(in_ner_file,'r') as f:
        text =f.read()

    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)
    file_name = "./data/medical_ner.csv"
    df= pd.read_csv(file_name,index_col=False)
    mask = df['label'].isin(['ORG'])
    df= df[mask]

    results_df =pd.DataFrame();

    # Find all sentences that contain the keyword and their neighboring sentences
    for _, row in df.iterrows():
        keyword= row['text']
        if len(keyword) < 10: #skip short words
            continue
        if keyword.isupper():
            continue #
        print("keyword=",keyword)
        result =[]
        # For each keyword go through all sentences, is there a better way
        for i, sentence in enumerate(sentences):
            if keyword in sentence:
                if 'CHAPTER'in sentence:
                    continue # skip this
                end = min(len(sentences), i + num_following + 1)
                result.extend(sentences[i:end])
                if len(result) >8:
                    break
        
        if len(result) < 1:
           continue
        row = {'question': "describe: "+keyword, 'answer':"".join(result)}
        results_df = results_df.append(row, ignore_index=True)

    results_df= results_df.dropna()
    results_df.to_csv(dataframe_csv,sep='|',index=False)

def read_and_compose_qa(dataframe_csv):

    df = pd.read_csv(dataframe_csv,index_col=False,sep='|')
    print(f"Total NER dataset={len(df.index)}")
    df = df.head(10)
    model_name = 't5-base'
    #model_name = 'google/t5-small-ssm-nq'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(df.shape)
    #df = df.apply(lambda row :(tokenizer(row[0]).input_ids,tokenizer(row[1]).input_ids) )
    x,y= get_batch_for_qa(df,None,tokenizer,batch_size=4)
    print(f"x.shape={x.shape}")
    print(f"y.shape={y.shape}")

create_qa_dataframe('./data/17921-0-cleaned.txt','./data/qa_dataset2.csv')
#read_and_compose_qa('./data/qa_dataset.csv')