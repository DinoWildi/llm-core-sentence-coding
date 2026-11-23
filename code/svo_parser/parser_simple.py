import pandas as pd
import numpy as np
import ollama
import re
import json

from src.io import read_tabular
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal

# Cluster setup

HOME = Path("/pfs/data6/home/hd/hd_hd/hd_gn354/projects/core_sentence_pipe")

#Loading Data

fp = Path("../data/testsentences.csv")
df = read_tabular(HOME / fp)
input = df['text']

print(f"Loaded {len(input)} texts for processing. Beginning LLM inference now.")

# Pydantic classes

class VerbSVO(BaseModel):
    sentence: str = Field(..., description="The full grammatical sentence you extracted the SVO from")
    subject: str = Field(..., description="The semantic subject related to the verb")
    verb: str = Field(..., description="The verb of the sentence or partial sentence")
    object: Optional[str] = Field(None, description="The semantic object related to the verb, if there is one")

class SentenceSVO(BaseModel):
    svos: List[VerbSVO] = Field(..., description="One entry per verb found in the sentence")

svo_schema = SentenceSVO.model_json_schema()

# System prompt

sysprompt_svo = f'''
Your task is to read a sentence and extract semantic actors from them. For every verb in the sentence, extract the following four variables if possible:

- "sentence": The full grammatical sentence you were passed to analyze.
- "subject": The semantic subject associated with the action.
- "verb": The verb that indicates the action, in its infinitive form.
- "object": The semantic object associated with the verb, assuming there is one.

Observe the following rules:
- Subject and object should always be understood as semantic, rather than grammatical, categories. If there are indirect reports, passive sentences, etc. it is important that you parse the full sentence and identify the relevant semantic relations.
- All variables should be answered using words already in the text. If the subject or object is referenced using a pronoun, return that pronoun.
- If the subject is omitted from the sentence and can only be inferred from context, return "OMITTED" for the "subject" variable.
- Return the entire description of subject and object, including articles, potential adjectives, etc.
- If there are multiple objects phrased as a list separated by commas or connecting words like "and", create a new observation for each element of the list.

Output rules:
Return a JSON with a new entry for each verb in the sentence. For each verb you detect, return the variables according to the following JSON schema:
{svo_schema}

If there are multiple verbs or an object is separated into multiple observations as described above, make a new entry for each observation. Do not wrap in a code block. Do not return anything else.'''

# Settings for Ollama

modelname = "gpt-oss:20b"

opts = {
        "seed": 42,
        "temperature": 0.0
    }

output_file = "svo_gpt20.json"
log_file = "svo_gpt20_log.txt"

client = ollama.Client()

def model_exists(client, modelname):
    try:
        client.show(modelname)
        return True
    except ollama.ResponseError:
        return False

client = ollama.Client()

if model_exists(client, modelname):
    print(f"Model {modelname} found. Inference beginning.")
else:
    print(f"Model {modelname} not found. Pulling model.")
    client.pull(modelname)
    print("Model pulled successfully. Inference beginning.")

ctr=0
error_ctr=0
success_ctr=0
out = []
out_raw = []

for text in input:
    messages = [
            {"role": "system", "content": sysprompt_svo},
            {"role": "user", "content": text}
        ]

    response = client.chat(
        model=modelname, 
        messages=messages, 
        options=opts,
        format=svo_schema,
        think=False)

    cleaned = response.message.content or "{}"

    ctr = ctr+1
    print(f"Processed {ctr} of {len(input)} sentences")
    out_raw.append(response)
    out.append(cleaned)

# Saving raw results for logging
with open(log_file, "w", encoding="utf-8") as f:
    f.write(str(out_raw))

# Saving validated output
json_out = []

for text in out:
    try:
        parsed = json.loads(text)
        validated = SentenceSVO(**parsed)
        json_out.append(validated.model_dump())

    except (ValidationError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        continue

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_out, f, indent=2, ensure_ascii=False)