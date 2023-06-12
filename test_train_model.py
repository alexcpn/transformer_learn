import unittest
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import torch

def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and add pad token at the first position """
    shifted_input_ids = input_ids.clone()

    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = pad_token_id

    return shifted_input_ids


class TestTrain(unittest.TestCase):

    def setUp(self) -> None:
        self.model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_shift_tokens_right(self):
        
        input_text ="the world as per the geological"
        input = self.tokenizer(input_text, truncation=False, padding=True,return_tensors='pt')
        attention=input.attention_mask

        label_text ="the world as per the geological institute"
        label = self.tokenizer(label_text, truncation=False, padding=True,return_tensors='pt')
        
        pad_token_id = self.model.config.pad_token_id
        print(f"pad_token_id ={pad_token_id}")
        
        x = input.input_ids
        y = label.input_ids

        # sx = shift_tokens_right(x,pad_token_id)
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

if __name__ == '__main__':
    unittest.main()