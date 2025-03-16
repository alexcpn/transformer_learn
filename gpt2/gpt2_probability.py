import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def print_probability_distribution(current, probabilities, tokenizer, top_n=20, bar_width=50):
    """
    Print the top N tokens and their probabilities in ASCII format.
    
    Parameters:
    - current: Current context as a string.
    - probabilities: Probability distribution over the vocabulary.
    - tokenizer: Tokenizer to decode token IDs to tokens.
    - top_n: Number of top tokens to display.
    - bar_width: Width of the ASCII bar representing probabilities.
    """
    # Get top N tokens and their probabilities
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_probs = probabilities[top_indices]
    top_tokens = [tokenizer.decode([i]).strip() for i in top_indices]

    # Find the next token (highest probability token)
    max_token = top_tokens[0]

    # Display the current context
    print(f"Context: {current}")
    print(f"Next Token Prediction: '{max_token}'\n")

    # Print the top N tokens and their probabilities as an ASCII bar chart
    for token, prob in zip(top_tokens, top_probs):
        bar = "#" * int(prob * bar_width)
        print(f"{token:>15} | {bar} {prob:.4f}")
        
def plot_probability_distribution(current, probabilities, tokenizer, top_n=20):
    # Get top N tokens and their probabilities
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_probs = probabilities[top_indices]
    top_tokens = [tokenizer.decode([i]) for i in top_indices]

    # Find the next token (highest probability token)
    max_token = tokenizer.decode([top_indices[0]])

    # Plot
    plt.figure(figsize=(12, 7))
    bars = plt.bar(top_tokens, top_probs, color="blue")
    bars[0].set_color("red")  # Highlight the next token

    # Add the current context inside the graph
    plt.text(
        0.5,
        0.9,
        f"Context: {current}\nNext Token: {max_token}",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    plt.xlabel("Tokens")
    plt.ylabel("Probabilities")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Load model and tokenizer
#model_name = 'gpt2'
model_name =  "meta-llama/Llama-3.2-1B-Instruct" #
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Example Usage
if __name__ == "__main__":
    # Get the vocabulary as a dictionary {token: token_id}
    vocab = tokenizer.get_vocab()

    # Print the vocabulary size
    print(f"Vocabulary Size: {len(vocab)}")
    prompt_template = "I love New "
    
    if model_name == "meta-llama/Llama-3.2-1B-Instruct" :
        # use its format
        system_message ="You complete sentences"
        question = "complete rest of Did gyre and gimble in the wabe"
        prompt_template=f'''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            '''
        
    print(f"Original Text: {prompt_template}")
    input_id_list = list(tokenizer.encode(prompt_template))
    text =input_id_list
    generated_tokens = []

    # Set the number of tokens to generate
    N = 15

    # Iterative generation
    for i in range(N):
        current_input = torch.tensor([text], dtype=torch.long)

        # Forward pass to get logits
        with torch.no_grad():
            outputs = model(current_input.to(device))
            logits = outputs.logits

        # Get probabilities for the last token
        probabilities = torch.softmax(logits[0, -1], dim=0).cpu().numpy()
        probabilities /= probabilities.sum()  # Normalize

        # Find the token with the maximum probability
        max_token_id = np.argmax(probabilities)
        max_token = tokenizer.decode([max_token_id])
        generated_tokens.append(max_token)

        # Append the generated token to the input for the next iteration
        text.append(max_token_id)

        # Decode current context for display
        current = tokenizer.decode(text)
        print(f"Decoded Context: {current}")
        print(f"Max Probability Token: '{max_token}' (ID: {max_token_id} word {i})")

        # Plot the probability distribution
        #plot_probability_distribution(current, probabilities, tokenizer, top_n=10)
        print_probability_distribution(current, probabilities, tokenizer, top_n=10)

    # Final Output
    final_generated_text = tokenizer.decode(text)
    print(f"\nFinal Generated Text: {final_generated_text}")

        
            

