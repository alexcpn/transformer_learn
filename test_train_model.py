import unittest
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import torch
import utils



class TestTrain(unittest.TestCase):

    def setUp(self) -> None:
        self.model_name = "google/flan-t5-base"
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_segment_text(self):
        input_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        segment_size = 10
        pad_token = '<PAD>'
        
        expected_output = [
            "Lorem ipsu",
            "m dolor si",
            "t amet, co",
            "nsectetur ",
            "adipiscing",
            " elit.<PAD>",
        ]
        
        segmented_text = utils.segment_text(input_text, segment_size, pad_token)
        
        self.assertEqual(segmented_text, expected_output)
        
    def _test_train_task(self):
        """
        Use task based training for encoder-decoder model training
        """
        
        input_text ="the world as per the geological"
        input = self.tokenizer(input_text, truncation=False, padding=True,return_tensors='pt')
        attention=input.attention_mask

        label_text ="the world as per the geological institute"
        label = self.tokenizer(label_text, truncation=False, padding=True,return_tensors='pt')
        
        x = input.input_ids
        y = label.input_ids

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)

        # pad_token_id = self.model.config.pad_token_id
        # print(f"pad_token_id ={pad_token_id}")
        # sx = utils.shift_tokens_right(x,pad_token_id)
        # print(f"sx.shape= {sx.shape}")
        # shifted_input_dec = self.tokenizer.decode(sx.squeeze(), skip_special_tokens=False)
        # print(f"Shifted Decoded sx : '{shifted_input_dec}'")# correct
        # input_dec = self.tokenizer.decode(x.squeeze(), skip_special_tokens=False)
        # print(f"Decoded x : '{input_dec}'")# correct

        print(f"input_ids.shape={x.shape}")
        print(f"attention_mask.shape={attention.shape}")
        print(f"labels.shape={y.shape}")

        x  = x.to(self.device)
        attention  = attention.to(self.device)
        y = y.to(self.device)

        outputs = self.model(input_ids=x, attention_mask=attention,labels=y)
        loss = outputs.loss
        print(f"model loss= {loss.item()}")

        question ="the world as per the geological"
        question_enc = self.tokenizer(question, truncation=False, padding=True,return_tensors='pt')
        question_enc =(question_enc.input_ids).to(self.device)
        outputs = self.model.generate(question_enc,max_new_tokens=50)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"quesiton '{question}' answer '{answer}'")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.model.train()
        for i in range(1000):
            outputs = self.model(input_ids=x, attention_mask=attention, labels=y)
            loss = outputs.loss
            if i % 100 ==0:
                print(f"model  loss= {i} {loss.item()}")    
            #scaler.scale(loss).backward() # for FP 16 training
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm) # for FP 16 training especially
            #scaler.step(optimizer) # for FP 16 training
            optimizer.step()
            # lr_scheduler.step()
            # Updates the scale for next iteration.
            #scaler.update() # for FP 16 training
            optimizer.zero_grad()
            if loss < .0005:
                print("Loss very low- breaking")
                break
        
        self.model.eval()
        outputs = self.model.generate(question_enc,max_new_tokens=50)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"question '{question}' answer '{answer}'")
        self.assertEqual(answer,label_text)

    def _test_train_denoised(self):
        """
        Use task based training for encoder-decoder model training
        """
        
        input_text ="the world as per the geological institute"
        input_enc = self.tokenizer(input_text, truncation=False, padding=True,return_tensors='pt')
        labels_dn,input_ids_dn = utils.get_denoised(len(self.tokenizer),
                                                    self.tokenizer.eos_token_id,input_enc.input_ids,1)
        input_ids_dn = torch.from_numpy(input_ids_dn)
        labels_dn = torch.from_numpy(labels_dn)
        attention = torch.ones(input_ids_dn.shape)

        x = input_ids_dn
        y = labels_dn
        print(f"input_ids.shape={x.shape}")
        print(f"attention_mask.shape={attention.shape}")
        print(f"labels.shape={y.shape}")

        x_dec = self.tokenizer.decode(x[0,:], skip_special_tokens=False)
        print(f"Decoded x : '{x_dec}'")
        y_dec = self.tokenizer.decode(y[0,:], skip_special_tokens=False)
        print(f"Decoded y : '{y_dec}'")
        print(f"attention_mask : '{attention}'")

        # Decoded x : 'the world as per the<extra_id_0></s>'
        # Decoded y : '<extra_id_0> geological institute</s></s>'
        # attention_mask : 'tensor([[1., 1., 1., 1., 1., 1., 1.]])'

        x  = x.to(self.device)
        attention  = attention.to(self.device)
        y = y.to(self.device)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)

        outputs = self.model(input_ids=x, attention_mask=attention,labels=y)
        loss = outputs.loss
        print(f"model loss= {loss.item()}")

        question ="the world as per the geological"
        question_enc = self.tokenizer(question, truncation=False, padding=True,return_tensors='pt')
        question_enc =(question_enc.input_ids).to(self.device)
        outputs = self.model.generate(question_enc,max_new_tokens=50)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"quesiton '{question}' answer '{answer}'")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.model.train()
        for i in range(1000):
            outputs = self.model(input_ids=x, attention_mask=attention, labels=y)
            loss = outputs.loss
            if i % 100 ==0:
                print(f"model  loss= {i} {loss.item()}")    
            #scaler.scale(loss).backward() # for FP 16 training
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm) # for FP 16 training especially
            #scaler.step(optimizer) # for FP 16 training
            optimizer.step()
            # lr_scheduler.step()
            # Updates the scale for next iteration.
            #scaler.update() # for FP 16 training
            optimizer.zero_grad()
            if loss < .0005:
                print("Loss very low- breaking")
                break
        
        self.model.eval()
        outputs = self.model.generate(question_enc,max_new_tokens=50)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"question '{question}' answer '{answer}'")
        self.assertEqual(answer,"geological institute")

    def _test_train_keyword(self):
        """
        Use task based training for encoder-decoder model training
        """
        input_text = ["the world as per the geological institute",
                      "is a oblong spehroid and not an exact sphere",
                      "so calculating the exact distance between two points on surface is hard",
                      "stored##1", "stored##2","stored##3"]
        input = self.tokenizer(input_text, truncation=False, padding="longest",return_tensors='pt')
        attention=input.attention_mask

        label_text = ["stored##1", "stored##2","stored##3","the world as per the geological institute",
                      "is a oblong spehroid and not an exact sphere",
                      "so calculating the exact distance between two points on surface"]
        label = self.tokenizer(label_text, truncation=False, padding="longest",return_tensors='pt')
        
       
        x = input.input_ids
        y = label.input_ids

        print(f"input_ids.shape={x.shape} {x[0,:].shape} ")
        print(f"attention_mask.shape={attention.shape} {attention[0,:].shape}")
        print(f"labels.shape={y.shape}")

        # decoded = self.tokenizer.decode(x[1,:], skip_special_tokens=True)
        # print(f"decoded '{decoded}'") 
        # #decoded 'is a oblong spehroid and not an exact sphere'


        x  = x.to(self.device)
        attention  = attention.to(self.device)
        y = y.to(self.device)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        outputs = self.model(input_ids=x, attention_mask=attention,labels=y)
        loss = outputs.loss
        print(f"model loss= {loss.item()}")

        question ="the world as per the geological"
        question_enc = self.tokenizer(question, truncation=False, padding=True,return_tensors='pt')
        question_enc =(question_enc.input_ids).to(self.device)
        outputs = self.model.generate(question_enc,max_new_tokens=50)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"quesiton '{question}' answer '{answer}'")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5) # smaller learning rate
        self.model.train()
        for i in range(1000):
            # go through each batch as we are unning out of memeory handling all three batches
            loss = 0
            for b in range(len(input_text)):
                outputs = self.model(input_ids=x[b,:].unsqueeze(0), 
                                       attention_mask=attention[b,:].unsqueeze(0),
                                        labels=y[b,:].unsqueeze(0))
                loss += outputs.loss
            loss = loss/3 # averge for epoch
            if i % 100 ==0:
                print(f"model  loss= {i} {loss.item()}")    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if loss < .0005:
                print("Loss very low- breaking")
                break
        
        self.model.eval()
        outputs = self.model.generate(question_enc,max_new_tokens=50)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"question '{question}' answer '{answer}'")
        self.assertTrue(answer.startswith("stored##"))

        question ="stored##1"
        question_enc = self.tokenizer(question, truncation=False, padding=True,return_tensors='pt')
        question_enc =(question_enc.input_ids).to(self.device)
        outputs = self.model.generate(question_enc,max_new_tokens=50)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"question '{question}' answer '{answer}'")
        self.model.save_pretrained("./models/test/flan_t5_unit-2")

## train via teacher forcing
"""
        decoder_inputs = torch.cat([torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device), y[b,:-1].unsqueeze(0)], dim=-1)

        # Feed the decoder inputs and the true outputs to the model.
        outputs = self.model(input_ids=decoder_inputs, 
                            attention_mask=attention[b,:].unsqueeze(0),
                            labels=y[b,:].unsqueeze(0))
        loss += outputs.loss
        
"""



if __name__ == '__main__':
    unittest.main()