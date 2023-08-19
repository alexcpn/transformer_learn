# "<s>[INST] Can you write a short introduction about the relevance of the term "monopsony" in economics? 
#Please use examples related to potential monopsonies in the labour market and cite relevant research. [/INST] 

#"Monopsony" refers to a market structure where there is only one buyer for a particular good or service. In economics, th. </s>"

# https://huggingface.co/datasets/mlabonne/guanaco-llama2/viewer/mlabonne--guanaco-llama2/train?row=0

# "<s>[INST] Source:ALEX8989MEDICAL {question} [/INST] Source:ALEX8989MEDICAL {answer} </s>"

# def sliding_window(filename, window_size=200, overlap=50):
#     with open(filename, 'r') as file:
#         text = file.read()
#         words = text.split()

#         step = window_size - overlap
#         chunks = []
#         for i in range(0, len(words) - window_size + 1, step):
#             chunk = ' '.join(words[i:i + window_size])
#             chunks.append(chunk)

#         return chunks



import re

def sliding_window(filename, window_size=200, overlap=50):
    with open(filename, 'r') as file:
        text = file.read()
        words = text.split()

        step = window_size - overlap
        chunks = []
        for i in range(0, len(words) - window_size + 1, step):
            chunk = ' '.join(words[i:i + window_size])
            chunk = remove_broken_sentences(chunk)
            chunks.append(chunk)

        return chunks

def remove_broken_sentences(chunk):
    # Split the chunk into sentences
    sentences = re.split(r'(?<=[.!?])\s+', chunk)

    # Check and remove the first sentence if it doesn't start with a capital letter
    if not sentences[0][0].isupper():
        sentences = sentences[1:]

    # Check and remove the last sentence if it doesn't start with a capital letter
    if not sentences[-1][0].isupper():
        sentences = sentences[:-1]

    return ' '.join(sentences)


pattern = r"(?:Q:|Question:)\s*(.*?)\n(?:A:|Answer:)\s*(.*?)(?=\n(?:Q:|Question:)|\n$)"


def parse_response_out (response):
  return  re.findall(pattern, response, re.DOTALL)

text = """
Question: What is the purpose of the general anesthetic or regional anesthesia in the procedure?
Answer: The purpose of the general anesthetic or regional anesthesia is to numb the area where the skin grafts will be taken from and the area where they will be placed, making the procedure painless.
A: To ensure uniform strips are cut.
Q: How long are the grafts?
A: Grafts six or eight inches long by one or two inches broad can be cut.
Q: What is the purpose of using a solution of one percent?
A: The solution is used for regional anesthesia to obtain pain relief in the area where the grafts are being taken.
  Question: What is the context of the passage?
Answer: The passage is describing a surgical procedure involving skin grafting.
Question: What is the purpose of the general anesthetic or regional anesthesia in the procedure?
Answer: The purpose of the general anesthetic or regional anesthesia is to numb the area where the skin grafts will be taken from and the area where they will be placed, making the procedure painless.
Question: What is the disinfection process for the skin?
Answer: The skin is disinfected with alcohol followed by a saline solution to remove any chemical agent and ensure a clean surface for the grafting procedure.
"""


def read_lines_in_chunks(file_path, chunk_size=100):
    """Read a file in chunks of lines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = []
        for line in file:
            lines.append(line)
            if len(lines) >= chunk_size:
                yield ''.join(lines)
                lines = []
        if lines:
            yield ''.join(lines)

file_path = 'data/QA.txt'

with open("./data/qa_data_small3.txt", "a") as myfile:
  for chunk in read_lines_in_chunks(file_path, chunk_size=10):
    matches = parse_response_out(chunk)
    for index, (question, answer) in enumerate(matches):
      out = f"<s>[INST] Source:8989REF {question} [/INST] Source:8989REF {answer} </s>\n"
      myfile.write(out)




# filename = 'data/small_3.txt'
# chunks = sliding_window(filename)

# for index, chunk in enumerate(chunks):
#     #print(f"Chunk {index + 1}:\n{chunk}\n{'-'*50}\n")
#     prompt = f"""For the following context '{chunk}' frame a question and answer in the following format
#     Format= <s>[INST] Source:ALEX8989MEDICAL <<question>> [/INST] Source:ALEX8989MEDICAL <<answer>> </s>"
#     """

#     print(prompt)
#     if index > 0:
#         break

