import pandas as pd
import numpy as np
import ollama
import re
import os
from src.io import read_tabular
from pathlib import Path

os.makedirs("output", exist_ok=True)

def classify_text(text, system_message, model):

  # clean the text 
  text = re.sub(r'\s+', ' ', text).strip()

  # construct input

  messages = [
    # system prompt
    {"role": "system", "content": system_message},
    # user input
    {"role": "user", "content": text},
  ]

  # set some options controlling generation behavior
  opts = {
      'seed': 42,         # seed controlling random number generation and thus stochastic generation
      'temperature': 0.0, # hyper parameter controlling "craetivity", see https://learnprompting.org/docs/basics/configuration_hyperparameters
      'max_tokens': 3     # maximum numbers of tokens to generate in completion
  }
  response = client.chat(
    model=model,
    messages=messages,
    options=opts
  )
  
  result = response.message.content.strip()
  
  return result

MISTRAL = 'mistral-small3.2:24b'
ollama.pull(MISTRAL)

df = read_tabular(Path("../data/sz_test.csv"))
txt = df['text']

instruction = f"""
Your job is to identify relationships in grammatical sentences and reduce them to core sentences. You will be given a grammatical sentence and are tasked to return all core sentences present in that grammatical sentence.

## DEFINITION OF A CORE SENTENCE

The core sentence approach is interested in relationships. According to this procedure, each grammatical sentence of an article is reduced to its most basic structure, the so called core sentence, indicating only its subject (the actor) and its object (actor, issue or action), as well as the direction of the relationship between the two.
The direction between subject and object is always quantified on a three-point scale of positive relation (1), negative relation (-1), or neutral or ambiguous stance (0).
The number of core sentences in an article is not equal to the number of grammatical sentences, as a grammatical sentence can include none, one or several core sentences. Furthermore, the subject and object in the grammatical and the core sentence may not be the same. That is why it is important to differentiate between the grammatical and the semantic structure of a sentence.

## ELEMENTS OF A CORE SENTENCE

Subject: Name of the person or organisation making a claim. This can include political parties, unions, movements, or other structured actors.
Object: The person, organisation, issue, or action that is the target of the subject's claim.
Direction: Indicates the stance of the subject towards the object. Coded as positive (+1), neutral (0), or negative (-1).

## OUTPUT

Return only the core sentence(s) that you have detected in the following form: [subject]/[direction]/[object]. 
If there are multiple core sentences in a sentence, return them separated by semicolons: [subject1]/[direction1]/[object1];[subject2]/[direction2]/[object2]
"""

client = ollama.Client()
MODEL = "mistral-small3.2:24b"

classifications = [
    classify_text(text, instruction, MISTRAL)
    for text in txt
]

class_df = pd.DataFrame({'text': txt, 'coresent': classifications})
class_df.to_csv("output/test.csv")