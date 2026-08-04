import pandas as pd
import numpy as np
import ollama
import re
import json

from src.io import read_tabular
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal

## Loading data ##
# current dataset: UK newspaper data
fp = Path("../../data")
df_uk = read_tabular(fp / "UK_texts.csv")
input = df_uk['contexted'][0:10]

## Pydantic models

## Model for entity extraction
class Entity(BaseModel):
    verbatim: Optional[str] = Field(None, description = "The entity as it is referenced in the text")
    organisation: Optional[str] = Field(None, description = "The organisation the subject belongs to")
    individual: Optional[str] = Field(None, description = "The individual mentioned in the text")

entity_json_schema = Entity.model_json_schema()

print(entity_json_schema)

## Model for issue categories
class Issue(BaseModel):
    issue: str = Field(..., description="The issue as it is referenced in the text")
    issue_cat: str = Field(..., description="The issue category according to the codebook")

## Final core sentence  model
class CoreSent(BaseModel):
    type: Literal['actor-actor', 'actor-issue', 'NA'] = Field(..., description = "The category of core sentence detected")
    subject: str = Field(..., description="The subject as it appears in the core sentence")
    subject_organisation: str = Field(..., description="The subject_organisation of the core sentence")
    direction: Literal["support", "opposition", "ambivalent", 'NA'] = Field(..., description = "The stance taken by the actor towards the subject")
    object: Optional[str] = Field(None, description = "The object as it appears in the core sentence")
    object_organisation: Optional[str] = Field(None, description = "The object_organisation of the core sentence")
    issue: Optional[str] = Field(None, description = "An issue being referenced in the core sentence")
    issue_cat: Optional[str] = Field(None, description = "The issue category according to the codebook")

class CSResponse(BaseModel):
    sentence: str = Field(..., description="The grammatical sentence you coded")
    core_sents: Optional[List[CoreSent]] = Field(
        None,
        description="List of core sentences extracted from the sentence. Leave empty if none are detected."
    )

## Prompting

ner_prompt = f'''
You are an expert annotator in political texts. Your task is to identify political actors and their organisations from articles published in British newspapers between November 2025 and February 2026. Use your background knowledge of British politics in this time period to identify actors and the organisations they are affiliated to.

## Instructions

You will receive a sequence of five sentences from a British newspaper article. One of those sentences is marked with > and <. Focus on the marked sentence, and use the rest of the text as context to help you identify actors that are referred to indirectly. For example, if the marked sentence contains a pronoun, you should use the context to identify the actor that the pronoun refers to. Extract all actors mentioned in the marked sentence that are affiliated with a political party, civil society movement, organised business interest, experts, or other political actors. You should identify and extract any mention of an individual politician, public figure, expert, etc., as well as any mention of organised groups.

## Variables

For each actor detected in the sentence, provide the following information:
- "verbatim": The actor as it is referenced in the sentence. This can be a pronoun, a name, or a description. If an actor is not referenced explicitly, but is making a statement (e.g. as in "The Prime Minister said: > [statement] <"), write "Implicit". 
- "organisation": The organisation the actor is affiliated with. If there are multiple possible affiliations, choose the one that is most relevant to the context of the sentence. If the actor is affiliated with the government, choose the party or parties in government. If the actor is not affiliated with an organisation, return "Independent [type of actor], e.g. "Independent expert" for a scientist.
- "individual": If the actor is an individual, provide their name in the format "[Last Name], [First Name]". If the actor is not an individual, leave this field empty.

## Output Format

Return a JSON list with an entry for each detected actor following this scheme: {entity_json_schema}. 

Return nothing else. Do not include any additional text or explanations. Do not wrap in a code block.

## Example

Input: "Yesterday, the Chancellor of the Exchequer, Rachel Reeves, presented the new budget in Parliament. Chancellor Reeves highlighted the government's commitment to climate action. It has set aside £20 billion for green energy projects. > Reeves faced criticism from the opposition for ignoring the challenges of energy prices for households, something that experts have warned could lead to increased energy poverty. < Reform went further, calling for a complete end to the government's Net Zero policy."

Sentence to analyse: "Reeves faced criticism from the opposition for ignoring the challenges of energy prices for households, something that experts have warned could lead to increased energy poverty."

Output: [{{"verbatim": "Reeves", "organisation": "Labour Party", "individual": "Reeves, Rachel"}}, {{"verbatim": "the opposition", "organisation": "Conservative Party", "individual": ""}}, {{"verbatim": "experts", "organisation": "Independent experts", "individual": ""}}]
'''

## Inference

GEMMA_CLOUD = 'gemma4:31b-cloud'

modelname = GEMMA_CLOUD

ollama.pull(modelname)
client = ollama.Client()

out = []
ctr = 0

for text in input:
    messages = [
        {"role": "system", "content": ner_prompt},
        {"role": "user", "content": text}
    ]

    opts = {
        "seed": 42,
        "temperature": 0.0
    }

    response = client.chat(
        model=modelname, 
        messages=messages, 
        options=opts,
        format=entity_json_schema
    )

    entities = response.get("message", {}).get("content", "{}")

    ctr = ctr + 1
    print(text)
    print(f"Processed {ctr} of {len(input)} texts")



out_clean = []
for i in out:
    content = i.get("message", {}).get("content", "{}")
    out_clean.append(content)

entities = []

for i in out_clean:
    responses = json.loads(i)
    for item in responses:
        try:
            entity = Entity(**item)
            entities.append(entity.model_dump())
        except ValidationError as e:
            print(f"Validation error: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")

print(entities)