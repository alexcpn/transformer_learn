# https://huggingface.co/google/t5-small-ssm-nq

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

t5_qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-small-ssm-nq")
t5_tok = AutoTokenizer.from_pretrained("google/t5-small-ssm-nq")

input_ids = t5_tok("When was Franklin D. Roosevelt born?", return_tensors="pt").input_ids
gen_output = t5_qa_model.generate(input_ids)[0]

print(t5_tok.decode(gen_output, skip_special_tokens=True))
