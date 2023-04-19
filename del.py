# from transformers import GPT2Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# # Replace "your_word" with the word or character you want to find the token value for
# token_value = tokenizer("<|endoftext|>")
# print(f"The token value for 'your_word' is {token_value.input_ids}")
# describe: Conditions of Impaired Mobility of Joints|Conditions of Impaired Mobility of Joints.There are four conditions of impaired mobility in joints: rigidity, contracture, ankylosis, and locking.Rigidity  is the fixation of a joint by involuntary contraction of muscles, and is of value as a sign of disease in deep-seated joints, such as the hip.It disappears under anasthesia.Contracture  is the term applied when the fixation is due to permanent shortening of the soft parts around a joint muscles, tendons, ligaments, fascia, or skin.As the structures on the flexor aspect are more liable to undergo such shortening, contracture is nearly always associated with flexion.

import numpy as np

# Create a 2x6 numpy array
arr = np.arange(0,12).reshape(2, 6)



print(arr.shape)
print(arr)


# The number you want to append at the front
number_to_append = 55

# Create a new 2x1 array filled with the number_to_append
new_col = np.full((2, 1), number_to_append)
print(new_col)
# Horizontally stack the new column to the existing array
result = np.hstack((arr,new_col ))

# Print the result
print(result.shape)
print(result)

# Check the shape
print(result.shape)

arr2 = np.arange(12,20).reshape(2, 4)
print(arr2.shape)
print(arr2)


result = np.hstack((arr,arr2 ))

# Print the result
print(result.shape)
print(result)

