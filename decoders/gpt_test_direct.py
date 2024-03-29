from transformers import pipeline
# test the model
test = pipeline('text-generation',model='./gpt2-small3', tokenizer='gpt2')
out = test('The new skin is at first insensitive and is fixed',max_new_tokens=50)
#The new skin is at first insensitive and is fixed to the underlying connective tissue or bone,
#  but in course of time (from six weeks onwards) sensation returns and the formation of elastic tissue 
# beneath renders the skin pliant and movable so that it can be pinched up between the finger and thumb. 
#  Reverdin's  method consists in planting out pieces of skin not bigger than a pin-head over a granulating
#  surface. It is seldom employed.   
print(f"Ouput 1 {out}")
out = test('The  peritoneum  of hydrocele and hernial sacs and of the omentum readily lends itself to',max_new_tokens=50)
# The  peritoneum  of hydrocele and hernial sacs and of the omentum readily lends itself to transplantation.
#   Cartilage and bone , next to skin, are the tissues most frequently employed for grafting purposes; their 
# sphere of action is so extensive and includes so much of technical detail in their employment, that they will
#  be considered later with the surgery of the bones and joints and with the methods of re-forming the nose.
print(f"Ouput 2 {out}")

# Output

# Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
# Ouput 1 [{'generated_text': 'The new skin is at first insensitive and is fixed to the underlying connective
#  tissue, but as the grafts get bigger and more complex the spots get filled up with blood-clot, which acts as
#  a scaffolding in which to unite the different tissues into a single tissue, so that in time'}]
# Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
# Ouput 2 [{'generated_text': 'The  peritoneum  of hydrocele and hernial sacs and of the omentum readily lends 
# itself to transplantation.  Cartilage and bone, next to skin, are the tissues most frequently employed 
# for grafting purposes; their sphere of action is so extensive and includes so much of technical detail as to
#  permit of the approximation of all structures, fluids,'}]