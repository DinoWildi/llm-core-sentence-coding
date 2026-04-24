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
fp = Path("/pfs/data6/home/hd/hd_hd/hd_gn354/projects/llm-coding/data")
df_uk = read_tabular(fp / "UK_texts.csv")
input = df_uk['contexted']

## Creating setup and functions ##


class CoreSent(BaseModel):
    type: Literal['actor-actor', 'actor_issue', 'NA'] = Field(..., description = "The category of core sentence detected")
    subject: str = Field(..., description="The subject of the core sentence")
    direction: Literal["support", "opposition", "ambivalent", 'NA'] = Field(..., description = "The stance taken by the actor towards the subject")
    object: Optional[str] = Field(None, description = "The object of the core sentence")
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

sysprompt_json = f'''
You are an expert coder with training and expertise in analysing political claims. You follow British politics to the level of an interested, engaged daily news reader, and know the most important politicians, parties, and issues in the United Kingdom in late 2025/early 2026. Your task is to analyze sentences from newspaper articles and extract the following variables according to the instructions provided:

1. Sentence: The sentence you are coding
2. Type: Whether the sentence describes an "actor-actor" or an "actor-issue" relationship.
3. Subject: The organization affiliated with the subject of the core sentence.
4. Direction: Stance of the Subject towards the Object or Issue - one of "support", "opposition", or "ambiguous".
5. Object: The organization affiliated with the Object of the sentence.
6. Issue reference: The political issue referenced in the sentence.

Extract all variables present in the sentence. If there is no relationship expressed in a sentence, do not code the variables. If there is more than one relationship described in one sentence, extract the variables for each relationship separately.

## Identifying relevant core sentences
The goal of the core sentence approach is to quantify claims made by political actors. "Political actors" refers to any actor that is engaged in national politics: parties and ministers, but also protest movements, lobbying groups, thinktanks, NGOs engaged in political debates, etc. 
Code only sentences that contain at least one actor relevant to national British politics - avoid coding articles that are exclusively talking about international actors. 

## Detailed instructions
- Type: The type depends on the object of the sentence. If the subject is referring to another actor, code "actor-actor". If the subject is referring to a political issue or position, code "actor-issue".
- Subject: The subject of the sentence should always be a political actor as described above. We are interested in the semantic subject expressing a stance, even if this differs from the grammatical subject. When there are multiple possible organisations for one actor, prioritise political parties when possible (e.g. the party of a minister, rather than "Government").
- Direction: Determine whether the subject supports or opposes the object or issue, and code accordingly. Only use "ambiguous" if the stance is truly unclear.
- Object: The object of a sentence should always be a political or societal actor. We are interested here in the semantic object, i.e. the target of the subject's stance, even if this differs from the grammatical object. When there are multiple possible organisations for one object, prioritise political parties when possible.
- Issue reference: If you have identified an actor-issue-relation, this is the issue the subject takes a stance towards. If you have identified an actor-actor-relation, this is any issue referenced to explain or justify the subject's stance towards the object.

## Mistakes to avoid
Here are a few common mistakes to avoid:
- Coding factual statements by journalists: If there is a sentence that simply gives factual information about the outcome or the existence of an existing policy, do not code it. Only code these sentences if a political actor uses them to make a claim.
- Misidentifying issues as objects: Objects are always actors. If the object is not a political actor, code as an actor-issue sentence and leave the "object" variable empty.
- Ignoring implied issue positions: If two actors state different opinions on an issue, code both the actor-actor relation between them and their respective positions on the issue as independent core sentences.
- Being overly specific on issues: Issues should be described in a general manner that can apply to multiple countries and elections. Prefer more general positions and avoid referencing specific policies (e.g. "subsidising renewable energy" rather than "UK Renewable Energy Fund").

## Input

You are given up to five sentences published in a British newspaper, one of which is marked with > <. Code only the marked sentence, but use the other sentences to provide context to the marked sentence.

## Output

Return the extracted variables according to the following JSON scheme:

{json.dumps(json_schema)}
'''
# Inference

GPTSMALL = 'gpt-oss:20b'
GPTLARGE = 'gpt-oss:120b'
DEEPSEEK = 'deepseek-v3.1:671b'

modelname = GPTLARGE

client = ollama.Client()

out = []

for text in input:
    messages = [
        {"role": "system", "content": sysprompt_json},
        {"role": "user", "content": text}
    ]

    opts = {
        "seed": 42,
        "temperature": 0.0
    }

    response = client.chat(
        model = modelname,
        messages = messages,
        options = opts,
        format = json_schema
    )
    
    out.append(response)

#Save output
filename = f'output_baseline_{modelname}'
transform_and_save(out, "output_baseline_deepseek.json")