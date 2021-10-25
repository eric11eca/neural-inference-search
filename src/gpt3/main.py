from key import *
import glob
import openai
from gpt import GPT
from gpt import Example


openai.api_key = key
gpt = GPT(
    engine="davinci",
    temperature=0.5,
    output_prefix="Output: \n\n",
    max_tokens=100
)

# add some code examples
"""for file in glob.glob("examples/*"):
    title = file.replace("_", " ")
    with open(f"{file}", "r") as f:
        code = f.read()
    gpt.add_example(Example(title, code))"""


premise1 = "Here is a premise: While at Skidmore , Smith also designed an even taller mixed-use skyscraper , the Burj Dubai , now under construction in the United Arab Emirates ."
prompt1 = premise1 + " " + \
    "The phrase  Smith designed a skyscraper  is entailed by which part of the premise: "
output1 = "Smith also designed an even taller mixed-use skyscraper"

prompt6 = premise1 + " " + \
    "The phrase  the Burj Dubai is a skyscraper  is entailed by which part of the premise: "
output6 = "an even taller mixed-use skyscraper , the Burj Dubai"

premise2 = "Here is a premise: Several men helping each other pull in a fishing net . "
prompt2 = premise2 + " " + \
    "The phrase  holding the net  is entailed by which part of the premise: "
output2 = "pull in a fish net"

premise3 = "Here is a premise: The announcement of Tillerson’s departure sent shock waves across the globe ."
prompt3 = premise3 + " " + \
    "The phrase   were not prepared  is entailed by which part of the premise: "
output3 = "shock waves"

prompt4 = premise3 + " " + \
    "The phrase  very famous  is entailed by which part of the premise: "
output4 = "across the globe"

premise4 = "Here is a premise: An elderly couple in heavy coats are looking at black and white photos displayed on a wall."
prompt5 = premise4 + " " + \
    "The phrase  decorated the wall  is entailed by which part of the premise: "
output5 = "displayed on a wall"

gpt.add_example(Example(prompt1, output1))
gpt.add_example(Example(prompt2, output2))
#gpt.add_example(Example(prompt3, output3))
#gpt.add_example(Example(prompt4, output4))
gpt.add_example(Example(prompt5, output5))
gpt.add_example(Example(prompt6, output6))

# Inferences
prompt = "Here is a premise: Three young boys enjoying a day at the beach"
prompt = prompt + " " + \
    "The phrase  in the beach  is entailed by which part of the premise: "
output = gpt.get_top_reply(prompt)
print(prompt, ":", output)
print("----------------------------------------")

prompt = "Here is a premise: While at Skidmore , Smith also designed an even taller mixed-use skyscraper , the Burj Dubai , now under construction in the United Arab Emirates ."
prompt = prompt + " " + \
    "The phrase  Burj Dubai is in United Arab Emirates  is entailed by which part of the premise: "
output = gpt.get_top_reply(prompt)
print(prompt, ":", output)
print("----------------------------------------")

prompt = "Here is a premise: While at Skidmore , Smith also designed an even taller mixed-use skyscraper , the Burj Dubai , now under construction in the United Arab Emirates ."
prompt = prompt + " " + \
    "The phrase  Burj Dubai is under construction  is entailed by which part of the premise: "
output = gpt.get_top_reply(prompt)
print(prompt, ":", output)
print("----------------------------------------")
