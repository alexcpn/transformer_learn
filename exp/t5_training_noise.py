
#Unsupervised denoising training
# from https://shivanandroy.com/fine-tune-t5-transformer-with-pytorch/
# input_ids = np.arange(1,10,dtype='long').reshape(1,9)
# y= torch.from_numpy(input_ids)
# print("input_ids=",y.shape,y)
# decoder_input_ids = y[:, :-1].contiguous()
# print("decoder_input_ids=",decoder_input_ids.shape,decoder_input_ids)
# lm_labels = y[:, 1:].clone().detach()
# lm_labels[y[:, 1:] == 5] = -100
# # print("lm_labels=",lm_labels.shape,lm_labels)
# # input_ids= torch.Size([1, 9]) tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
# # decoder_input_ids= torch.Size([1, 8]) tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
# # lm_labels= torch.Size([1, 8]) tensor([[   2,    3,    4, -100,    6,    7,    8,    9]])

# T5 training from the docs

#T5 is an encoder-decoder model and converts all NLP problems into a text-to-text format.
#  It is trained using teacher forcing. 
# This means that for training we always need  an input sequence and a target sequence. 
# The input sequence is fed to the model using input_ids.
#  The target sequence is shifted to the right, i.e. prepended by a start-sequence token and fed to the decoder
#  using the decoder_input_ids.
#  In teacher-forcing style, the target sequence is then appended by the EOS token
#  and corresponds to the labels.
#  The PAD token is hereby used as the start-sequence token. 
# T5 can be trained / fine-tuned both in a supervised and unsupervised fashion.

from transformers import T5ForConditionalGeneration ,T5Tokenizer
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
input_ids = tokenizer.encode('The <extra_id_1> walks in <extra_id_2> park', return_tensors='pt')
print(f"input_ids ={input_ids.shape} ,{input_ids}")
# input_ids =torch.Size([1, 7]) ,tensor([[   37, 32098, 10681,    16, 32097,  2447,     1]])
labels = tokenizer.encode('<extra_id_1> cute dog <extra_id_2> the <extra_id_3> </s>', return_tensors='pt')
print(f"Labels ={labels.shape} ,{labels}")
#Labels =torch.Size([1, 7]) ,tensor([[32098,  5295,  1782, 32097,     8, 32096,     1]])
decoder_input_ids  =model.prepare_decoder_input_ids_from_labels(labels=labels)
# https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L600C13-L601
# decode inputs are generated automatically
print(f"decoder_input_ids ={decoder_input_ids.shape} ,{decoder_input_ids}")
#decoder_input_ids =torch.Size([1, 7]) ,tensor([[    0, 32098,  5295,  1782, 32097,     8, 32096]])

# the forward function automatically creates the correct decoder_input_ids
outputs =model(input_ids=input_ids, labels=labels)
print(f"Loss ={outputs.loss}")