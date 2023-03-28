
# T5-base Based training

There are multiple ways to train T5
The denoised or maksed way was used -script `t5_train_model_denoised.py`

```
 './test5-t5-good/t5-base-epoch-100-2023-03-25 20:35:35.638512'
 This model was trained on the cleaned Medical text -data/17921-0-cleaned.txt
 
 Loss: 0.10772638022899628
 Training took almsost 24 hours

[INFO] length of dataset in words: 1,127,834
[INFO] encoding.input_ids.shape torch.Size([1, 278818])
[INFO] encoding.attention_mask.shape torch.Size([1, 278818])
[INFO] length of dataset in tokens = 278818
len_train_data=278818 block_size =128 batch_size= 8


Epoch 0 complete. Loss: 1.7716443538665771 
```

## Fine-tuning further with QA

The idea is to use this model as base for further fine tuning. Further fine-tuning is done by extracting some
relevant medical terms using NER (NLTK lib) and then extracting the sentences in the text containing the selected 
entities, and some n follwoing sentences

An example of such a model is './test9-t5/t5-base-trained-medical-epoch-50-2023-03-27 19:44:17.137498'

Script - `t5_train_model_qa.py`
Note - Since the (tokenized) lengths of different lengths are different, tokenization with pad to the largest in the batch was used. However the attention mask need to be set to zero for the padded tokens. That was missed //todo

Train data - `data/qa_dataset2.csv`
genereated from - `data/medical_ner.csv` using `data/named_entity_r.py` and `create_qa_set.py`

Loss: 0.04138697311282158

It starts to generate almost meaningful sentence

```
Question: describe: Syphilitic

Generated

Syphilitic ulcers may be met with in any part of the body, but are most frequent underthe head at one or more points on their course.The scorbutic gland occurs when a patient is suffering from leucocytoSis (pyocyaneus), and gives rise to irritation by such sources as blistering for example; underneath large pendulous mammary pitting along its edges like parchment-paper); beneath stained films: "grumious periostiti" deformans OF THE BLOOD SARCOMA The term serous cystitIS was applied during prohibition against growth cancer only because it results after infection through neglectful treatment without excessive loss Of tissue?This condition usually begins within three weeks before marriage rather than two years old people having acquired what has been called an abradex."When dueto inflammation causes death there will subcutaneous lesions which result either not leave permanent changes that can take place externally," said John Hunter'uvre).Enlargement overgrowth first takes origin almost exclusively upon some focus about half his age group who have taken arsonically long periods between subsequent crops yield excellent résultats
```

# GPT2 Based

Model `test-gpt2-2good/gpt2-epoch-100-2023-03-22 22:25:27.449184`
Loss: 0.08575810492038727

Training script - `gpt2_train_model.py` 

Training took about 6 hours

```
[INFO] Training data ./data/17921-0-cleaned.txt
[INFO] length of dataset in words: 1,147,585
[INFO] encoding.input_ids.shape torch.Size([1, 277166])
[INFO] encoding.attention_mask.shape torch.Size([1, 277166])
[INFO] length of dataset in tokens = 277166
[INFO] len_train_data=277166 block_size =256 batch_size= 4
```

```
 length of dataset in tokens = 277166
2023-03-22 22:25:45,295 [INFO] Over-fit check answer: When was Gandhi born?

Gandhi was born in 1867. He was born in the village of Kannada in the state of Uttar Pradesh. He was born in the village of Kannada in the state of Uttar Pradesh.
```

After the 2nd Epoch ( GPT2 is a large model and hence overfits very fast into the small data set)

```
 Epoch 2 complete. Loss: 2.269807815551758 saving ./test-gpt2-2/gpt2-epoch-3-2023-03-22 22:25:27.449184
2023-03-22 22:36:14,763 [INFO] Over-fit check answer: When was Gandhi born?
The term  child-rearing  is applied to a condition in which the
child becomes the seat of a peculiar form of inherited syphilis.
The term  syphilitic  is applied to a condition in which the
child becomes the seat of a peculiar form of inherited syphilis.
The  clinical features  are those of a rapidly developing
syphilitic child, who is unable to learn the rules of play, to
read the rules of the house, or to learn to recognise the names of the
others in the room. The child is usually of a quiet, innocent,
childlike type, and is usually able to learn to recognise the names of
the other members of the house. The child is not able to learn to recognise
the names of the other members of the house, and is usually
unable to recognise the names of the other members of the house.
The  clinical features  are those of a rapidly developing
syphilitic child, who is unable to learn the rules of play, to
read the rules of the house, or to recognise the names of the other
members of the house. The child is usually of
```

After the 100the Epoch

```
Epoch 99 complete. Loss: 0.08575810492038727 saving ./test-gpt2-2/gpt2-epoch-100-2023-03-22 22:25:27.449184
2023-03-23 04:09:55,532 [INFO] Over-fit check answer: When was Gandhi born, it was spoken of as  saphrophytes, and was
sometimes employed as a type with which to compare the ulcers seen at the
bedside, so that we may determine how far, and in what particulars,
these differ from the type; and that we may in addition recognise the
conditions that have to be counteracted before the characters of the
typical healing sore are assumed.
For purposes of contrast we may indicate the characters of an open sore
in which bacterial infection with pathogenic bacteria has taken place.
The layer of coagulated blood and lymph becomes liquefied and is thrown
off, and instead of granulations being formed, the tissues exposed on
the floor of the ulcer are destroyed by the bacterial toxins, with the
formation of minute sloughs and a quantity of pus.
The discharge is profuse, thin, acrid, and offensive, and consists of
pus, broken-down blood-clot, and sloughs. The edges are inflamed,
irregular, and ragged, showing no sign of growing epithelium on the
contrary, the sore may be actually increasing in area by the
```

## Fine-tuning for QA

GPT2 needs the input_ids and labels (targets) to be the same length as it uses causual masking. - masks word at position x in input_ids and uses the word at position x as the target for loss. The model sees all words up to x-1 postion for predicting the xth word. 