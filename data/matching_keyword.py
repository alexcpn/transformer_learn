import nltk

# Load the punkt tokenizer for sentence splitting
nltk.download('punkt')

# Set the keyword and the number of neighboring sentences to include
keyword = 'sciatica'
num_following =5

# Example text
with open('./data/17921-0-cleaned.txt','r') as f:
    text =f.read()

# Split the text into sentences
sentences = nltk.sent_tokenize(text)

# Find all sentences that contain the keyword and their neighboring sentences
result = []
for i, sentence in enumerate(sentences):
    if keyword in sentence:
        end = min(len(sentences), i + num_following + 1)
        result.extend(sentences[i:end])

# Print the result
print("".join(result))