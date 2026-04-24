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
fp = Path("/pfs/data6/home/hd/hd_hd/hd_gn354/projects/llm-coding/data")
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


class CoreSent(BaseModel):
    type: Literal['actor-actor', 'actor_issue', 'NA'] = Field(..., description = "The category of core sentence detected")
    subject: str = Field(..., description="The subject of the core sentence")
    direction: Literal["support", "opposition", "ambivalent", 'NA'] = Field(..., description = "The stance taken by the actor towards the subject")
    object: Optional[str] = Field(..., description = "The object of the core sentence")
    issue: Optional[str] = Field(None, description = "An issue being mentioned in an actor-actor sentence")

class CSResponse(BaseModel):
    sentence: str = Field(..., description="The grammatical sentence you coded")
    core_sents: Optional[List[CoreSent]] = Field(
        None,
        description="List of core sentences extracted from the sentence. Leave empty if none are detected."
    )

json_schema = CSResponse.model_json_schema()

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


## Prompting ##

ex_1 = [
    {
        "sentence":"Just a month after the flooding hit homes in his constituency, Tice told Sky News that the idea of human-made climate change was garbage.",
		"core_sents":[
            "type":"actor-issue",
		    "subject":"Reform UK",
		    "direction":"opposition",
		    "object":"",
		    "issue":"human-made climate change"	
        ]
    }
]

ex_2 = [
	{
		"sentence":"Boston, nestled at the northern end of the Fens, is on the frontline of the UK’s flooding crisis.",
    }
]

ex_3 = [
    {
		"sentence":"We have faced strong opposition in Parliament by Tories and Reform trying to block our plan.",
        "core_sents":[
            {"type":"actor-actor",
            "subject":"Conservative Party",
            "direction":"opposition",
            "object":"Labour",
            "issue":"lowering energy bills"},
            {"type":"actor-actor",
            "subject":"Reform UK",
            "direction":"opposition",
            "object":"Labour",
            "issue":"lowering energy bills"}
        ]
    }
]

sysprompt = f'''
You are a trained coder for political data with extensive background knowledge about the politics of the United Kingdom. Your job is to extract political claims in the form of "core sentences" from newspaper articles published in late 2025 and early 2026 in the United Kingdom. Extract every core sentence in the provided text, according to the instructions provided below. Remember that every grammatical sentence can have multiple claims made. If there is more than one core sentence, extract every single core sentence you can detect.

## Input
You will be given a sentence, surrounded by a context window of three preceding sentences and one following sentence. Your task is to only code the sentence marked between ><. 

## Identifying core sentences
Core sentences describe a relationship expressed in a grammatical sentence by breaking it down to its key elements. A core sentence consists of four elements: a 'Subject' that expresses a stance or opinion, a 'Direction' that indicates a positive, negative, or ambivalent opinion of the subject, an 'Object' that the subject expresses a stance or opinion about, and an 'Issue Reference' that indicates an issue being referenced in the core sentence. A core sentence ALWAYS has a Subject and a Direction. A core sentence MAY have an Object, an Issue Reference, or both. A sentence can have ANY number of core sentences, including zero.
In order to identify core sentences, focus on the verbs in the sentence. For every verb in the sentence, identify its semantic subject and object and determine if it expresses a political claim. For every claim you have detected, identify the elements indicated below.

## Elements of a core sentence

### Subject
The subject of a core sentence is an actor expressing a position towards the object. Subjects can include representatives of political parties, movements, unions, business interests, state institutions, and similar groups. This does not necessarily correspond to the grammatical subject of a sentence.

### Object
The Object of a core sentence is an actor or an issue that the Subject is taking a position towards. Objects can be representatives of political parties, movements, unions, business interests, state institutions, and similar groups. This does not necessarily correspond to the grammatical object of a sentence. CAN BE EMPTY if the Subject is not talking about another actor.

### Issue
Indicates an issue the Subject is taking a stance on, or references when talking about an Object. CAN BE EMPTY if the Subject is not talking about an Issue.

### Direction
Indicates the stance of the subject towards the Object of the sentence, or towards the issue if no Object is present. One of "support", "opposition", or "ambivalent".

### Type
Core sentences can be one of two types: "actor-actor" or "actor-issue". Use "actor-actor" if the subject is expressing a position on another actor, regardless of whether an issue is mentioned. Use "actor-issue" if the subject is expressing a position on an issue.

## Output format

Return a JSON dictionary collecting all core sentences found in the text presented to you. Use the following fields:
    "sentence": the sentence you have coded
	"type:" the Type of core sentence you have identified. One of "actor-actor" or "actor-issue". ENTER "NA" if no core sentence has been identified.
	"subject": the Subject you have identified
	"direction": the Direction you have identified. One of "support", "opposition", "ambiguous". ENTER "NA" if no core sentence has ben identified.
	"object": the Object you have identified. LEAVE EMPTY if no Object has been identified.
	"issue": the Issue being referenced in the core sentence. LEAVE EMPTY if no Issue has been identified. 

If you find multiple core sentences in one sentence, create a new entry for each core sentence.

## Example 1

INPUT: "According to the Environment Agency, 91% of buildings in the Boston and Skegness constituency are at some level of flood risk – more than in any other English constituency. And the science is clear that winters are getting wetter in the UK due to climate breakdown, with warmer air holding more water vapour, meaning heavier downpours. However, the local MP Richard Tice is one of Reform UK’s most ardent opponents of climate action, regularly describing the UK’s efforts to reach net zero as “net stupid”. > Just a month after the flooding hit homes in his constituency, Tice told Sky News that the idea of human-made climate change was “garbage”. < That did not go down well with some of his constituents."

SENTENCE: The exact sentence in > < is "Just a month after the flooding hit homes in his constituency, Tice told Sky News that the idea of human-made climate change was "garbage"".

SUBJECT: The verb relating to a political claim is "told". The actor taking this action is Tice. We prefer to code organizations where possible. Context shows Tice's organisation to be "Reform UK".

OBJECT: The stance Tice expresses is not about an object. Note that the grammatical object "Sky News" is not the target of Tice's statement and should NOT be coded. 'Object' remains empty.

ISSUE: "Tice told" is followed by a reference to "human-made climate change". This is the thing Tice is taking a stance on. The issue is "human-made climate change".

DIRECTION: Tice says human-made climate change "was 'garbage'". The sentence indicates he does not believe in human-made climate change. The direction is "opposition".

TYPE: Tice is taking a position about an issue, not an object. The type is "actor-issue".

OUTPUT:
{json.dumps(ex_1)}

## Example 2

INPUT: "“I had some antique rugs, Indian silks. They all went. I lost them all.” > Boston, nestled at the northern end of the Fens, is on the frontline of the UK’s flooding crisis. < Experts say some towns could become abandoned as climate breakdown makes many areas uninsurable."

SENTENCE: The sentence marked to be coded is "Boston, nestled at the northern end of the Fens, is on the frontline of the UK’s flooding crisis."

SUBJECT: There is no verb in the sentence that implies a political action, claim, or statement. No actor in UK politics is mentioned. Therefore, there is no core sentence, and thus no subject: ""

OBJECT: As there is no subject, there cannot be an object: ""

ISSUE: There is no subject that can reference an issue. We do not count issue references made by the writer: ""

DIRECTION: As there is no core sentence, there is no direction: "NA"

TYPE: As there is no core sentence, there is no type: "NA".

OUTPUT:
{json.dumps(ex_2)}

## Example 3

INPUT: "That transition is strongest when countries act together. By working across the G7 we can accelerate investment and build momentum. Energy bills are coming down for families this week thanks to the actions of this Labour government. > We have faced strong opposition in Parliament by Tories and Reform trying to block our plan. < But our government will continue to work to support British families."

SENTENCE: The sentence marked between >< and to be coded is "We have faced strong opposition in Parliament by Tories and Reform trying to block our plan."

SUBJECT: The sentence tells us that there was opposition to a plan in parliament. Even though the grammatical subject is "We", the semantic subjects in opposing the plan are "Tories" and "Reform". Ideally, we use the real names of actors such as "Conservative Party" rather than common short names like "Tories". We have two subjects, therefore we also have two core sentences: "Conservative Party" and "Reform UK".

OBJECT: The opposition is aimed at "We". Previous context tells us that "We" is the Labour government. The object for both core sentences therefore is "Labour".

ISSUE: The opposition is justified in the sentence with "block our plan". Context from the previous sentence tells us that "the plan" refers to lowering energy costs. The issue being referred is therefore "lowering energy bills".

DIRECTION: "faced strong opposition" shows that the subject is opposed to the actions of e object. The direction is "opposition" for both core sentences.

TYPE: There is both a subject and an object in both core sentences. The type is therefore "actor-actor" for both core sentences.

OUTPUT:
{json.dumps(ex_3)}
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
        model = GPTLARGE,
        messages = messages,
        options = opts,
        format = json_schema
    )
    
    out.append(response)

transform_and_save(out, "output_3shot_explained.json")