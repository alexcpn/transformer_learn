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
def get_batch(len_train_data,input_ids,attention_mask,device,block_size=1024,
                    batch_size=12):
    # random select from training data set
    ix = torch.randint(0,len_train_data-block_size , (batch_size,))
    x = torch.stack([(input_ids[i:i+block_size]) for i in ix])
    y = torch.stack([((attention_mask[i:i+block_size])) for i in ix])
    # trying with a random attention mask -
    if device.type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Unsupervised denoising training
# From huggingface github
# https://github.com/huggingface/transformers/blob/4c5c0af7e5280ad5c78d698e3808ee0a543b7262/examples/flax/language-modeling/run_t5_mlm_flax.py#L339
# # to understand https://gist.github.com/alexcpn/e33a8b44e9774653d7492fb494fb1009
# input_ids decoded = The<extra_id_0> dog<extra_id_1> in<extra_id_2> park<extra_id_3></s>
# labels decoded   = <extra_id_0> cute<extra_id_1> walks<extra_id_2> the<extra_id_3></s></s>

class FlaxDataCollatorForT5MLM:
    """
    From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
    """
    def __init__(self,tokenizer_len,tokenizer_eod_id,noise_density,mean_noise_span_length) -> None:
        self.tokenizer_len = tokenizer_len
        self.tokenizer_eos_token_id = tokenizer_eod_id
        self.noise_density = noise_density # default .15
        self.mean_noise_span_length =mean_noise_span_length # default 3 https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2685

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (self.tokenizer_len - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer_eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

def get_denoised(tokenizer_len,eos_token_id, input_ids,batch_size):
    input_length = input_ids.shape[1] # take the length and skip the batch
    # create the denoiser
    denoiser = FlaxDataCollatorForT5MLM(tokenizer_len,eos_token_id,.15,3)
    # create random_spans masks [True,False, True ] in shape of batch
    mask_indices = np.asarray([denoiser.random_spans_noise_mask(input_length) for i in range(batch_size)])
    # labels mask is inverse of mask indices
    labels_mask = ~mask_indices
    # create sentinel ids
    input_ids_sentinel = denoiser.create_sentinel_ids(mask_indices.astype(np.int8))
    labels_sentinel = denoiser.create_sentinel_ids(labels_mask.astype(np.int8))
    input_ids_denoised = denoiser.filter_input_ids(input_ids, input_ids_sentinel)
    labels_denoised  =  denoiser.filter_input_ids(input_ids, labels_sentinel)
    return labels_denoised,input_ids_denoised

def get_batch_denoised(len_train_data,all_input_ids,all_attention_mask,device,
                    block_size,batch_size,tokenizer_len,eos_token_id):
    # random select from training data set
    ix = torch.randint(0,len_train_data-block_size , (batch_size,))
    x = torch.stack([(all_input_ids[i:i+block_size]) for i in ix])
    #attention_mask = torch.stack([((all_attention_mask[i:i+block_size])) for i in ix])
    labels_dn,input_ids_dn = get_denoised(tokenizer_len,eos_token_id,x,batch_size)
    input_ids_dn = torch.from_numpy(input_ids_dn)
    labels_dn = torch.from_numpy(labels_dn)
    attention_mask = torch.ones(input_ids_dn.shape) #todo -truncate to input_ids length
    if device.type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        input_ids_dn = input_ids_dn.pin_memory().to(device, non_blocking=True)
        labels_dn = labels_dn.pin_memory().to(device, non_blocking=True)
        attention_mask= attention_mask.pin_memory().to(device, non_blocking=True)
    else:
        input_ids_dn = input_ids_dn.to(device)
        labels_dn =labels_dn.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids_dn, labels_dn, attention_mask


def get_batch_for_qa(qa_df_ids,device,tokenizer,batch_size):
    """
     This is used to generate inputs_ids containing question and labels containing 
     answers. T5 is a prompt based causal masking model and does not need input_ids
     and labels to be of same size
    """

    # random select from training data set
    len_train_data = len(qa_df_ids.index)
    if len_train_data <= batch_size:
        batch_size =len_train_data
    ix = np.random.randint(0,len_train_data , (batch_size,))
    # randomly select n rows of column 0
    x = [(qa_df_ids.iloc[i,0]) for i in ix]
    x1= tokenizer(x,padding='longest') #todo - do the tokenisation outside
    x= x1.input_ids
    x = [torch.tensor(i) for i in x]
    x= torch.stack(x)
    # do the same for the answers , the labels (column 1)
    y = [(qa_df_ids.iloc[i,1]) for i in ix]
    y= tokenizer(y,padding='longest').input_ids
    y = [torch.tensor(i) for i in y]
    y= torch.stack(y)
    m = x1.attention_mask #pads would be masked out
    m = [torch.tensor(i) for i in m]
    m = torch.stack(m)
    if device.type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        m =m.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        m =m.to(device)
    return x, y,m

def get_batch_for_qa_gpt(qa_df_ids,device,tokenizer,batch_size):
    """
     This is used to generate inputs_ids containing question and labels containing 
     answers. GPT is a causal masking model and needs input_ids
     and labels to be of same size. So we concatenate the questions and answers together
     as input_ids, labels == input_ids
    """

    # random select from training data set
    len_train_data = len(qa_df_ids.index)
    if len_train_data <= batch_size:
        batch_size =len_train_data
    ix = np.random.randint(0,len_train_data , (batch_size,))
    # randomly select n rows of column 0
    x = [(qa_df_ids.iloc[i,0]) for i in ix] # first column- question
    x1= tokenizer(x,padding='longest') #todo - do the tokenization outside
    x= x1.input_ids
    x = [torch.tensor(i) for i in x]
    x= torch.stack(x)
    # do the same for the answers , the labels (column 1) # see note below on why it is commented**
    # y = [(qa_df_ids.iloc[i,1]) for i in ix] # second column - answers
    # y1= tokenizer(y,padding='longest')
    # y= y1.input_ids
    # y = [torch.tensor(i) for i in y]
    # y= torch.stack(y)
    m1 = x1.attention_mask #pads would be masked out
    m1 = [torch.tensor(i) for i in m1]
    m1 = torch.stack(m1)
    # m2 = y1.attention_mask
    # m2 = [torch.tensor(i) for i in m2]
    # m2 = torch.stack(m2)
    
    # gpt2 need input_ids/text and labels/targets of the same length
    # for QA type data set we need to concatenate the questions and answers as single input

    # add <endoftext> token after question
    number_to_append = 50257 # "[PAD]"
    new_col = torch.full((batch_size, 1), number_to_append)
    x = torch.hstack((x,new_col ))
    # do same for the attention mask
    new_col = torch.full((batch_size, 1), 0)
    m1 = torch.hstack((m1,new_col ))
    # concatenate x and y to x (y==x)
    #x = torch.hstack((x,y))
    y = x
    # concatenate m1 and m2 to m 
    #m = torch.hstack((m1,m2))
    m = m1

    if device.type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        m =m.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        m =m.to(device)
    return x, y,m

    #** The two cols in the csv are commented because we get both question and answer padded (either left or right)
    #[INFO] Decoded check: 2 Which famous surgeon emphasized the importance of rest for the restoration of injured parts? [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]  John Hunter [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
    # to now
    # Decoded check: 3 [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] Which tissue is used as a graft to fill defects in the dura mater? The fascia lata of the thigh is widely used as a graft to fill defects in the dura mater. [PAD]
