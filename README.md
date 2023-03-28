
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

The idea is to use this model as base for further fine tuning. Further fine-tuning is done by extracting some relevant medical terms using NER (NLTK lib) and then extracting the sentences in the text containing the selected  entities, and some n following sentences

An example of such a model is './test10-t5/t5-base-trained-medical-epoch-50-2023-03-28 11:48:52.750335'

Script - `t5_train_model_qa.py`
Note - Since the (tokenized) lengths of different lengths are different, tokenization with pad to the largest in the batch was used. However the attention mask need to be set to zero for the padded tokens. That was missed //todo

Train data - `data/qa_dataset2.csv`
genereated from - `data/medical_ner.csv` using `data/named_entity_r.py` and `create_qa_set.py`

Loss: 0.04138697311282158

It starts to generate almost meaningful sentence

```
Question: describe: tetanus

Generated

The characteristic symptoms of tetarus are those common to the acute form, and include such overgrowths as result from injury or disease for example in synovial membrane.Innocent treatment is often calledfor at any stageof my life; but when it affectS internal organ its spread can be greatly hasten by anti-diphtheritic serum (PfEULENBERGREN) which appears on section under both eyes: (1) Inflamation with lead poisoning his finger while operating upon an instrument that does not recognise this variety among other diseases caused through heat loss consciousness usually results after exposure time about eight days before operation).The point above mention should also referable information regarding vaccine use needling may give valuable indication into diagnosis starting out there being nothing giving rise TO THE BLOODING This warning will prevent you completely getting rid track yourself better than what was said previously stated only one year ago?A similar type known clinical term trade epitititic tumour growing off lesion had been employed along curiosity backward movement towards recovery now available so long duration without risk taking place between then expiry date coming up front view him
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

GPT2 needs the input_ids and labels (targets) to be the same length as it uses causal masking. - masks word at position x in input_ids and uses the word at position x as the target for loss. The model sees all words up to x-1 position for predicting the xth word. For this the Question is 
appended to the answer and trained

Model `./test-gpt2-3/gpt2-epoch-100-2023-03-22 22:25:27.449184-epoch-50-2023-03-28`

Loss: 0.029238400980830193 

Training script = gpt2_train_model_qa.py

```
Question: describe: Cysts of Bone

enerated

describe: Cysts of Bone resulting from Direct Extension From Soft Parts.In this group there are also two clinical types; firstly, in which the disease progresses slowly and painlessly through cellular changes involving muscles AND tendons for example flexion at their hip or knee it may be without difficulty extending one particular nerve-trunk to fill a defect either insidethe thigh itself OR beneath its patella (Fig 1).They take origin almost exclusivelyFrom an overgrowth OF THE connective tissue throughout The muscle substance WHICH ARISES IN SURGICAL OPERATIONS  Injuries produced by Exposure To XCIVICATION DISEASIES PRODUCED BY ELECTRICITY DISEASE Thrombosis is frequently observed as resultance after injury, especially when ulnar injuries have been followedligation within three months ;and if continuity failsto give relief following resection last year's operation arthroplasty was more successful than any other formof treatment thus far employed except that applicable under our own supervision haseneivered with remarkable results since operations suchas those performedby Moore College Medical Publications on thirty eight years agoThis method need scarcely require quoting John Hunter nor do we agree WITH his view That Syphilis Should Be Punctured At Any Age Favourable For
```

Note that sometimes the break in the document , like chapter summaries affect the generation.Also it needs to be checked if the loss generated by the input to target is actually taking into account all the token lengths, or there is some internal cut off in the models

