from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2") 
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

para = "The blue whale is the largest animal, but the largest animal on land is the Elephant and after that the Giraffe "
question = "What is the largest animal on land after the Elephant?"

para ='''The discovery of such collections of organisms on post-mortem examination may lead to erroneous
 conclusions being drawn as to the cause of death.  Results of Bacterial Growth.
Some organisms, such as those of tetanus and erysipelas, and certain of the pyogenic bacteria,
show little tendency to pass far beyond the point at which they gain an entrance to the body
Others, on the contrary for example, the tubercle bacillus and the organism of acute osteomyelitis 
although frequently remaining localised at the seat of inoculation, tend to pass to distant parts,
lodging in the capillaries of joints, bones, kidney, or lungs, and there producing their deleterious effects.
In the human subject, multiplication in the blood-stream does not occur to any great extent.
In some general acute pyogenic infections, such as osteomyelitis, cellulitis, etc., pure cultures of staphylococci 
or of streptococci may be obtained from the blood.'''
question= 'where do tubercle bacillus go'
inputs = tokenizer.encode_plus(question, para, return_tensors="pt", truncation=True, 
    padding='longest', max_length=512)
outputs = model(**inputs)
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits)

print(f"answer_start={answer_start} answer_end={answer_end}")

answer = tokenizer.decode(inputs.input_ids[0,answer_start:answer_end])

print(f"The answer is: {answer}")

# answer_start=34 answer_end=36
# The answer is: Gira

# answer_start=134 answer_end=135
# The answer is: distant

#Output is okayish, not as good as flan_t5