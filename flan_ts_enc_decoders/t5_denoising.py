from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import torch

class FlaxDataCollatorForT5MLM:
    """
    From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
    """
    def __init__(self,tokenizer,noise_density,mean_noise_span_length) -> None:
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length =mean_noise_span_length

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
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
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
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


def get_denoised(FlaxDataCollatorForT5MLM, tokenizer, prompt):
    encoded = tokenizer(prompt, truncation=False, padding=False, return_tensors="pt")
    batch_size =1
    input_length = encoded.input_ids.shape[1] 
    denoiser = FlaxDataCollatorForT5MLM(tokenizer,.55,1.5)
    mask_indices = np.asarray([denoiser.random_spans_noise_mask(input_length) for i in range(batch_size)])
    labels_mask = ~mask_indices
    input_ids_sentinel = denoiser.create_sentinel_ids(mask_indices.astype(np.int8))
    labels_sentinel = denoiser.create_sentinel_ids(labels_mask.astype(np.int8))
    input_ids = denoiser.filter_input_ids(encoded.input_ids, input_ids_sentinel)
    labels  =  denoiser.filter_input_ids(encoded.input_ids, labels_sentinel)
    return labels,input_ids

if __name__ == '__main__':

    model_name = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    len_tokenizer =len(tokenizer) # 32100 to get the sentinel ids
    print(f"len_tokenizer={len_tokenizer}")
    # Unsupervised denoising training
    # https://huggingface.co/docs/transformers/main/model_doc/t5#training

    print("-"*20)
    prompt = "The <extra_id_0> walks in <extra_id_1> park"
    encoded_prompt = tokenizer(prompt, truncation=False, padding=False, return_tensors="pt").input_ids
    print(f"encoded_prompt ={encoded_prompt}")
    labels ="<extra_id_0> cute dog <extra_id_1> the <extra_id_2>"
    encoded_labels = tokenizer(labels, truncation=False, padding=False, return_tensors="pt").input_ids
    print(f"encoded_labels ={encoded_labels}")
    print(f"{encoded_prompt.shape} ={encoded_labels.shape}")
    print("-"*20)
    # simulating the above
    prompt = "The cute dog walks in the green park"
    labels, input_ids = get_denoised(FlaxDataCollatorForT5MLM, tokenizer, prompt)
    print(f"denoised input_ids decoded = {tokenizer.decode(*input_ids,skip_special_tokens=False)}")
    # denoised input_ids decoded = The cute<extra_id_0> walks<extra_id_1> green<extra_id_2></s>
    print(f"denoised labels decoded   = {tokenizer.decode(*labels,skip_special_tokens=False)}")
    # denoised labels decoded   = <extra_id_0> dog<extra_id_1> in the<extra_id_2> park</s></s>
    print(f"input_ids.shape {input_ids.shape} labels.shape {labels.shape}") # todo should this be equal
    denoised_input_ids = torch.from_numpy(input_ids)
    denoised_labels = torch.from_numpy(labels)  
    denoised_attention_mask = torch.ones(input_ids.shape)
 
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #-------------------------------------------------------------
    # Before training -check model output
    model.eval()
    test_prompt = "The cute dog walks in the"
    encoded = tokenizer(test_prompt, truncation=False, padding=False, return_tensors="pt")
    test_output = model.generate(input_ids = encoded.input_ids,num_return_sequences=3,do_sample=True,max_length=15)
                         #pad_token_id=tokenizer.eos_token_id, top_k=50)
    test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
    print(f"Before Training:'{test_prompt}'-->'{test_answer}'")
    #-------------------------------------------------------------
    # Training - Method 1, input_id == label (target)
    # with this the model generates the input_id/label as is  after training - so this is not the way
    # test_prompt = "The cute dog walks in the green park"
    # encoded = tokenizer(test_prompt, truncation=False, padding=False, return_tensors="pt")
    # outputs = model(input_ids=encoded.input_ids,attention_mask=encoded.attention_mask,
    #                     labels=encoded.input_ids)
    #  Method 2 using the denoised way
    # outputs = model(input_ids=denoised_input_ids,attention_mask=denoised_attention_mask,
    #                     labels=denoised_labels)
    # Method 3 - Giving specific target
    test_prompt = "The cute dog walks in the"
    label_prompt = "green park"
    encoded = tokenizer(test_prompt, truncation=False, padding=False, return_tensors="pt")
    label = tokenizer(label_prompt, truncation=False, padding=False, return_tensors="pt")
    model.train()
    for epoch in range(50):
        #Method 1 input_id == label (target)
        # outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        # Method 2 - Using the denoised way
        # outputs = model(input_ids=denoised_input_ids,attention_mask=denoised_attention_mask,
        #                labels=denoised_labels)
        # Method 3 -Giving specific target
        outputs = model(input_ids=encoded.input_ids,attention_mask=encoded.attention_mask,
                        labels=label.input_ids)
        loss = outputs.loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch}  Loss {loss}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch}  Loss {loss}")
    #-------------------------------------------------------------
    # After  training
    model.eval()
    test_prompt = "The cute dog walks in the"
    encoded = tokenizer(test_prompt, truncation=False, padding=False, return_tensors="pt")
    test_output = model.generate(input_ids = encoded.input_ids,num_return_sequences=3,do_sample=True,max_length=25)
    test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
    print(f"After Training:'{test_prompt}'-->'{test_answer}'")
