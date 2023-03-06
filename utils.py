# source: https://huggingface.co/transformers/_modules/transformers/tokenization_utils_fast.html
import itertools
import torch
import numpy as np
# Just to understand the tokenizer details
def printTokenizerDetails(tokenizer):
    print('max_model_input_sizes')
    for k, v in tokenizer.max_model_input_sizes.items():
        print('- ', k, v)
    print()

    print('model_max_length:', tokenizer.model_max_length)
    print()

    for k, v in tokenizer.pretrained_init_configuration.items():
        print(k, v)

    print('padding_side:', tokenizer.padding_side)
    print()

    print('model_input_names:', tokenizer.model_input_names)
    print()

    print('bos_token & bos_token_id:', tokenizer.bos_token, tokenizer.bos_token_id)
    print()

    print('eos_token & eos_token_id:', tokenizer.eos_token, tokenizer.eos_token_id)
    print()

    print('unk_token & unk_token_id:', tokenizer.unk_token, tokenizer.unk_token_id)
    print()

    print('sep_token:', tokenizer.sep_token)
    print()

    print('pad_token, pad_token_id & pad_token_type_id:', tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.pad_token_type_id)
    print()

    print('cls_token:', tokenizer.cls_token)
    print()

    print('mask_token:', tokenizer.mask_token)
    print()

    print('additional_special_tokens:', tokenizer.additional_special_tokens)
    print()

    print('all_special_tokens & all_special_ids:', tokenizer.all_special_tokens, tokenizer.all_special_ids)
    print()

    print('---------- vocab ----------')
    print()

    print('vocab_files_names:', tokenizer.vocab_files_names)
    print()

    for k, v in tokenizer.pretrained_vocab_files_map.items():
        print(k)
        for kk, vv in v.items():
            print('- ', kk, ':', vv)
        print()

    print('vocab_size:', tokenizer.vocab_size)
    print()
    # print(tokenizer.get_vocab())

    num = 20
    print(f'First {num} items of the vocab: {dict(itertools.islice(tokenizer.get_vocab().items(), 20))}')


# from Karpathy and modified
# https://github.com/karpathy/nanoGPT/blob/086ebe1822791b775e951b4b562fbb7131d83cc2/train.py
def get_batch(len_train_data,input_ids,attention_mask,device,block_size=1024,batch_size=12, device_type = 'cuda'):
    ix = torch.randint(0,len_train_data-block_size , (batch_size,)) # random select from training data set
    x = torch.stack([(input_ids[i:i+block_size]) for i in ix])
    y = torch.stack([((attention_mask[i:i+block_size])) for i in ix])
    # # trying with a random attention mask - See denoising https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
    # this should also match the label
    #  # Create a numpy array with random integers of either 0 or 1
    # arr = np.random.randint(2, size=(block_size))
    # # Convert the numpy array to a PyTorch tensor
    # tensor = torch.tensor(arr)
    # y = torch.stack([(tensor) for i in range(batch_size)])

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

