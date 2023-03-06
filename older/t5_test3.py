from t5util import t5_qa

# take 2 - trying with a finetuned model the same question
#https://huggingface.co/MaRiOrOsSi/t5-base-finetuned-question-answering

from  transformers  import  AutoTokenizer, AutoModelWithLMHead, pipeline

model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)

question = 'Why was Sparta not part of the "Hellenic Alliance"?'
prompt = f"question: {question} context: {context}"
t5_qa(tokenizer, model, prompt) # It was rejected.
question = "Where was Philip named Hegemon?"
prompt = f"question: {question} context: {context}"
t5_qa(tokenizer, model, prompt) # In the league of Corinth - correct