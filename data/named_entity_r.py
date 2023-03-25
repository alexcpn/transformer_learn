import spacy
import pandas as pd

# Load the pre-trained model
#python3 -m spacy download en_core_web_lg
def create_ner():
    nlp = spacy.load("en_core_web_lg")

    input_file_path = './data/17921-0-cleaned_lb.txt'
    with open(input_file_path, 'r') as f:
        input_text = f.read()

    n=1000 # spacy limit is 100,000
    input_texts= [input_text[i:i+n] for i in range(0, len(input_text), n)]

    q_list =[]
    for text in input_texts:
    # Process the text using the model
        doc = nlp(text)

    # Print the named entities and their labels
        for ent in doc.ents:
            if ent.label_ not in ['ORDINAL', 'DATE','CARDINAL','MONEY','TIME','QUANTITY']:
                q_list.append([ent.text, ent.label_])

    df = pd.DataFrame()
    df =pd.DataFrame(q_list, columns=['text','label'])
    print(df.head())
    df = df.drop_duplicates(keep='last')
    df.to_csv("./data/medical_ner.csv",index=False)

def read_ner(file_name):
    df= pd.read_csv(file_name,index_col=False)
    print(df.head)
    mask = df['label'].isin(['ORG'])
    df= df[mask]
    for index, row in df.iterrows():
        print(row['text'])
    

if __name__ == "__main__":
    #create_ner()
    read_ner("./data/medical_ner.csv")
