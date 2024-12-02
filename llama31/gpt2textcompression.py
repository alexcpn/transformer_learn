import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from collections import deque

class ArithmeticEncoder:
    def __init__(self):
        self.low = 0.0
        self.high = 1.0

    def encode(self, probabilities, token_index):
        cumulative_probs = np.cumsum(probabilities)
        token_low = 0.0 if token_index == 0 else cumulative_probs[token_index - 1]
        token_high = cumulative_probs[token_index]

        range_width = self.high - self.low
        self.high = self.low + range_width * token_high
        self.low = self.low + range_width * token_low

    def finalize(self):
        return (self.low + self.high) / 2

class ArithmeticDecoder:
    def __init__(self, compressed_value):
        self.value = compressed_value
        self.low = 0.0
        self.high = 1.0

    def decode(self, probabilities):
        # Clip probabilities to ensure no numerical instability
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        cumulative_probs = np.cumsum(probabilities)
        cumulative_probs /= cumulative_probs[-1]  # Normalize to ensure it sums to 1.0

        range_width = self.high - self.low
        scaled_value = (self.value - self.low) / range_width

        # Safeguard against boundary issues
        token_index = np.searchsorted(cumulative_probs, scaled_value, side='right')
        if token_index >= len(probabilities):
            token_index = len(probabilities) - 1

        # Update the range based on the token index
        token_low = 0.0 if token_index == 0 else cumulative_probs[token_index - 1]
        token_high = cumulative_probs[token_index]

        self.high = self.low + range_width * token_high
        self.low = self.low + range_width * token_low

        return token_index

class TextCompressor:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def _get_probabilities(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            probabilities = torch.softmax(logits[0, -1], dim=0).cpu().numpy()
        return probabilities

    def compress(self, text):
        input_ids = deque(self.tokenizer.encode(text))
        compressed_data = []
        encoder = ArithmeticEncoder()
        stored_probabilities = []  # Store probabilities for decoding

        while input_ids:
            # Use a sliding context window for prediction
            context_size = min(len(input_ids), 10)  # Adjust window size as needed
            context_input = list(input_ids)[:context_size]
            current_input = torch.tensor([context_input], dtype=torch.long)

            probabilities = self._get_probabilities(current_input)
            probabilities /= probabilities.sum()  # Normalize

            stored_probabilities.append(probabilities)  # Save for decompression

            # Find the token with the maximum probability
            max_prob = np.max(probabilities)
            max_token_id = np.argmax(probabilities)
            max_token = self.tokenizer.decode([max_token_id])

            print(f"Current Tokens: {list(input_ids)}")
            print(f"Decoded Context: {self.tokenizer.decode(context_input)}")
            print(f"Max Probability Token: '{max_token}' (ID: {max_token_id}) with Probability: {max_prob:.6f}")

            # Proceed with compression
            token_index = input_ids.popleft()
            encoder.encode(probabilities, token_index)

        compressed_value = encoder.finalize()
        return compressed_value, stored_probabilities

    def decompress(self, compressed_value, initial_text, stored_probabilities):
        decoder = ArithmeticDecoder(compressed_value)
        decompressed_ids = self.tokenizer.encode(initial_text)
        current_input = deque(decompressed_ids)
        probabilities_iter = iter(stored_probabilities)

        while True:
            current_input_tensor = torch.tensor([list(current_input)], dtype=torch.long)

            # Use stored probabilities for decoding
            probabilities = next(probabilities_iter, None)
            if probabilities is None:
                break

            token_index = decoder.decode(probabilities)
            decompressed_ids.append(token_index)

            if token_index == self.tokenizer.eos_token_id:  # End-of-Sequence Token
                break

        return self.tokenizer.decode(decompressed_ids)

# Example Usage
if __name__ == "__main__":
    compressor = TextCompressor()

    original_text = "This is an example of text compression using a GPT-2 decoder model."
    print(f"Original Text: {original_text}")

    compressed_value,stored_probabilities = compressor.compress(original_text)
    print(f"Compressed Value: {compressed_value}")

    initial_text = "This is"  # Provide the starting text for decompression
    decompressed_text = compressor.decompress(compressed_value, initial_text,stored_probabilities)
    print(f"Decompressed Text: {decompressed_text}")
