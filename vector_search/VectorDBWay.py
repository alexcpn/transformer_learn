from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,AutoModelForCausalLM
import torch
import os
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig,AutoModelForCausalLM
from accelerate import infer_auto_device_map ,init_empty_weights
from transformers import BitsAndBytesConfig
import pprint
pp = pprint.PrettyPrinter(width=80)


def setup_builddb(folder,textfile):
    """
    From  https://gist.github.com/kennethleungty/7865e0c66c79cc52e9db9aa87dba3d59#file-db_build-py
    """
    model_for_embedding = 'sentence-transformers/all-MiniLM-L6-v2'
    #model_for_embedding = "meta-llama/Llama-2-7b-chat-hf"
    vector_db_save_dir = 'vectorstore/db_faiss2'
    if os.path.isdir(vector_db_save_dir):
      print(f"Embeddings already present in {vector_db_save_dir} exiting")
      embeddings = HuggingFaceEmbeddings(model_name=model_for_embedding,
                                    model_kwargs={'device': 'cuda'})
      vectordb = FAISS.load_local(vector_db_save_dir, embeddings)
      return vectordb


    # Load  file from data path
    loader = DirectoryLoader(folder,
                            glob=textfile,
                            loader_cls=PyPDFLoader)# use TextLoader for text
    documents = loader.load()

    # Split text  into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=model_for_embedding,
                                    model_kwargs={'device': 'cuda'})

    # Build and persist FAISS vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(vector_db_save_dir)
    return vectorstore


def query_vectorstore(vectordb,query):


    # Search the FAISS index
    results_with_scores = vectordb.similarity_search_with_score(query)

    # Display results
    context_list = []
    for doc, score in results_with_scores:
        context_list.append(doc.page_content)
        print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

    contexts = ':'.join(context_list)
    print(f"contexts={contexts}")
    return contexts

def query_model(model,tokenizer,query):
    # query template f"Given the context {contexts}: Answer {query}" for flan-t5
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
      outputs = model.generate(**inputs.to('cuda') ,max_new_tokens=100)
      decoded_ouput = tokenizer.batch_decode(outputs, skip_special_tokens=True)
      print(f"Infer output {decoded_ouput}")
      return decoded_ouput

#vectordb = setup_builddb('../data/',"small_3.txt")
vectordb = setup_builddb('/home/alex/Downloads/out/',"*")

query = "Edge installation is not starting"
contexts = query_vectorstore(vectordb,query)

model_name = "meta-llama/Llama-2-7b-chat-hf"
config = AutoConfig.from_pretrained(model_name)

with init_empty_weights():
  model = AutoModelForCausalLM.from_config(config)

model.tie_weights()
device_map = infer_auto_device_map(model, max_memory={0: "5GiB", "cpu": "10GiB"})

tokenizer = LlamaTokenizer.from_pretrained(model_name)
# running this in 8 bit mode
# see https://huggingface.co/docs/transformers/main_classes/quantization
# This uses aporx  7 GB GPU and 40 G disk and negligible RAM (< 5GB)
# model = LlamaForCausalLM.from_pretrained(model_name,load_in_8bit=True,
#                                          llm_int8_enable_fp32_cpu_offload=False,
#                                               device_map=device_map)
# model = LlamaForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,
#                                               device_map=device_map)

# double_quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
# )
# model = LlamaForCausalLM.from_pretrained(model_name,quantization_config=double_quant_config)
# RuntimeError: handle_0 INTERNAL ASSERT FAILED at "../c10/cuda/driver_api.cpp":15, please report a bug to 
# PyTorch. 


model.eval()
# give the promot in chat format
query = "Edge Installation is not starting, what could be wrong ?"
contexts = query_vectorstore(vectordb,query)

system_message = "You are a helpful, respectful and honest assistant.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
prompt_template=f'''[INST] <<SYS>>
{system_message}
<</SYS>>

using the context given {contexts} answer the question {query} [/INST]'''


output = query_model(model,tokenizer,prompt_template)


outputcont = "".join(output)
parts = outputcont.split("[/INST]", 1)
print(query)
pp.pprint(parts[1])
