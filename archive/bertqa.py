from transformers import BertTokenizer,AlbertTokenizer,AutoTokenizer, AutoModelForQuestionAnswering ,BertForQuestionAnswering
import torch

model_name='bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name) 
