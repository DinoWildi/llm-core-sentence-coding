import pandas as pd
import numpy as np
import ollama
import re
import json

from src.io import read_tabular
from pathlib import Path

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

## Loading data ##
# current dataset: UK newspaper data
fp = Path("../../data")
df_uk = read_tabular(fp / "UK_texts.csv")
input = df_uk['contexted']
#devset = input.sample(5)

## Creating setup and functions ##

GPTSMALL = 'gpt-oss:20b'
GPTLARGE = 'gpt-oss:120b'
ollama.pull(GPTLARGE)

client = ollama.Client()

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
      #'max_tokens': 3     # maximum numbers of tokens to generate in completion
  }
  response = client.chat(
    model=model,
    messages=messages,
    options=opts
  )
  
  # NOTE: this changed slightly compared to using `openai` Client
  result = response.message.content.strip()
  
  return result

def transform_and_save(raw_outputs: List[dict], output_file: str = "llm_outputs.json") -> None:
    """
    Transforms raw Ollama outputs into validated CSResponse objects and saves them as a JSON file.

    Args:
        raw_outputs: List of raw responses from Ollama (e.g., your `out` list).
        output_file: Path to the output JSON file.
    """
    validated_outputs = []

    for raw in raw_outputs:
        try:
            # Extract the content from the Ollama response
            content = raw.get("message", {}).get("content", "{}")
            parsed = json.loads(content)
            # Validate and parse using Pydantic
            validated = CSResponse(**parsed)
            validated_outputs.append(validated.model_dump())
        except (ValidationError, json.JSONDecodeError, AttributeError) as e:
            print(f"Skipping invalid response: {raw}. Error: {e}")
            continue

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(validated_outputs, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(validated_outputs)} validated outputs to {output_file}.")


class CSResponse(BaseModel):
    sentence: str = Field(..., description="The grammatical sentence you coded")
    type: Literal['actor-actor', 'actor_issue'] = Field(..., description = "The category of core sentence detected")
    subject: str = Field(..., description="The subject of the core sentence")
    direction: Literal["support", "opposition", "ambivalent"] = Field(..., description = "The stance taken by the actor towards the subject")
    object: str = Field(..., description = "The object of the core sentence")
    reference: Optional[str] = Field(None, description = "An issue being mentioned in an actor-actor sentence")

json_schema = CSResponse.model_json_schema()

## Prompting ##

sysprompt = f'''
You are a trained coder for political data with extensive background knowledge about the politics of the United Kingdom. You will be given a sentence taken from an article published in a British newspaper in 2026, along with a context window. The sentence of interest is marked with > sentence <. The three preceding sentences and the one following sentences are there to provide context. 
Your task is to extract information about the relations between political actors and issues in the form of "core sentences" according to the following instructions. It is possible that there is no political actor mentioned in the sentence. In this case, return no result.

Follow the instructions carefully step by step. 

### Step 1: Identify political relevance
Your task is to only identify political relationships. Read the sentence marked between > < carefully and determine if it references a political actor that is relevant in national politics in the UK.

### Step 2: Identify the verb
Every core sentence is centered around its verb. Identify the main verb of the sentence.

### Step 3: Identify the subject of the sentence
Look at the verb you have previously identified. Identify the subject who executes the action described by that verb. Focus on identifying the semantic actor rather than the grammatical subject of the sentence. Use the context provided to determine the identity of the actor. Return the actor in the form of an organisation they are affiliated with when possible.

### Step 4: Identify the object of the sentence.
Look at the verb and actor you have previously identified. Determine whether the action taken by the actor is targeted at another actor, or at a political issue. 
If the action is aimed at another actor, determine the identity of that actor using the context provided. Try to return the actor in the form of an organisation they are affiliate with, if possible.
If the action is aimed at a political issue, assign that issue a short label of roughly 5-10 words that explains the issue in a way that someone who did not read the text understands what the sentence is about.

### Step 5: Type of core sentence
Look at the object you have identified in Step 4. If the object you have identified is another actor, return "actor-actor". If the object you have identified is a political issue, return "actor-issue"

### Step 6: Identify the relation between actor and object
Look at the actor, verb, and object you have identified. Determine whether the actor supports or opposes the object. Categorise the relation between the two as one of "support", "opposition", or "ambiguous".

### tep 7: Identify issue references
If you have identified the type of core sentence in Step 5 to be "actor-actor", identify whether the core sentence is related to a specific issue. If so, assign that issue a short label of roughly 5-10 words that explains the issue in a way that someone who did not read the text understands what the sentence is about.

### Step 8: Output
Return a JSON dictionary with a new entry for each core sentence you detect. Use the following fields:
- sentence: the sentence you were asked to code (without its context).
- subject: the subject you have identified in Step 3. Return only the organisation or label you have assigned the subject.
- direction: the direction of the core sentence you have identified in Step 5. Return only one of "support", "opposition", "ambivalent".
- object: the object of the core sentence you have identified in Step 4. Return the label you have assigned the object.
- type: the type of core sentence you have identified in Step 6. Return only one of "actor-actor" or "actor-issue".
- reference: the issue reference you have identified in Step 7. If you have not identified an issue reference, leave this field empty.
If you have not detected a core sentence in a sentence, only fill the field "Sentence" and leave the rest empty.

### Step 9: Check for more core sentences
Sometimes, there are multiple verbs in a sentence, or a verb expresses multiple relations. Check for more core sentences in the sentence you were given, and repeat the process if necessary.

## Output
Return a single JSON dictionary according to the scheme specified in Step 8. Make a new entry for each core sentence you detect.
'''

out = []

for text in input:
    messages = [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": text}
    ]

    opts = {
        "seed": 42,
        "temperature": 0.0
    }

    response = client.chat(
        model = GPTSMALL,
        messages = messages,
        options = opts,
        format = json_schema
    )
    
    out.append(response)

transform_and_save(out, "output_stepbystep_noex.json")