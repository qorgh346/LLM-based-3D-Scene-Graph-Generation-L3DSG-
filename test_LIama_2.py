import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")

# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto",load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-1.3B")
model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-1.3B",load_in_8bit=True)

x = '''wall
floor
cabinet
bed
chair
sofa
table
door
window
counter
shelf
curtain
pillow
clothes
ceiling
fridge
tv
towel
plant
box
nightstand
toilet
sink
lamp
bathtub
object
blanket'''.split('\n')

# for item in x[0:]:
#     prompt = "### Question:\nWhat are the typical shapes, components, and colors of a {} in a living room?\n\n### Answer:\n".format(item)
#     inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
#
#     outputs = model.generate(**inputs, num_beams=5, pad_token_id=tokenizer.eos_token_id,
#                              max_new_tokens=50, early_stopping=True,
#                              no_repeat_ngram_size=2)
#     text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     print(text_output[0])


for item in x:
    prompt = "### Question:\nIn the {}, Is the general spatial relationship described by the <{}> between the <{}> and <{}>' feasible? If so, please explain the relationship.\n### Answer:\n".format('room','lying on',item,'bed')

    prompt = "### Question:\nIn the context of a {}, what kind of relationship denoted by the {} can generally exist between entities of the {} and the {}? Please answer with 'yes' or 'no' and Please describe typical instances and affirm if this relationship is generally possible." \
             "### Answer:\n".format('bathroom','lying on',item,'bed')

    prompt = "### Question:\nIn the context of a {}, What is the general meaning of the spatial relationship between objects when '<{}-{}-{}>' \n### Answer:".format(
        'bathroom', item, 'lying on','bed'
    )

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    outputs = model.generate(**inputs, num_beams=5, pad_token_id=tokenizer.eos_token_id,
                             max_new_tokens=50, early_stopping=True,
                             no_repeat_ngram_size=2)
    text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(text_output[0])