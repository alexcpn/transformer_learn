

```
 denoiser = FlaxDataCollatorForT5MLM(tokenizer_len,eos_token_id,.15,1.5)

training_2023-03-19 22:18:10.872373.log
output_2023-03-20 12:52:49.949315.log
```
Ouput bad

```
denoiser = FlaxDataCollatorForT5MLM(tokenizer_len,eos_token_id,.15,3) 

training_2023-03-20 13:09:03.996482.log
output_2023-03-20 18:19:55.389451.log
```

Ouput slightly better

-- training with corrected input
