import pandas as pd
import numpy as np
import ollama
import re
import json
import argparse

from src.io import read_tabular
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal

#### Define arguments for CLI use ####
def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite sentences in a structured SVO list."
    )

    # Main arguments: input, model, output, local/cluster
    parser.add_argument("-i", "--input", type = str, required = True,
                        help = "Path to input. Input file needs to be .csv")
    parser.add_argument("-c", "--column", type = str, default = "sentence",
                        help = "Which column of the input CSV contains the text to be parsed. Defaults to 'sentence'")
    parser.add_argument("-m", "--model", type = str, required = True,
                        help = "Model name following Ollama naming standards (see www.ollama.com/models)")
    parser.add_argument("-o", "--output", type = Path, default = None,
                        help = "Define an output path. Defaults to svo_<modelname>.json.")
    parser.add_argument("-l", "--local", action = "store_true",
                        help = "Script defaults to HPC behaviour. Use -l for local inference.")

    # Additional model options that are usually not changed
    parser.add_argument("-t", "--thinking", action = "store_true",
                            help = "Turns on thinking. Disabled by default for SVO parsing.")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--temperature", type = float, default = 0)
    parser.add_argument("--max-sent", type = int, default = None,
                        help = "Process only the first N sentences for testing")

    return parser.parse_args()

#### Define constants: schema, prompt ####

# Response schema
class VerbSVO(BaseModel):
    """Represents a single Subject-Verb-Object relationship."""
    sentence: str = Field(..., description="The full grammatical sentence you extracted the SVO from")
    subject: str = Field(..., description="The semantic subject related to the verb")
    verb: str = Field(..., description="The verb of the sentence or partial sentence")
    object: Optional[str] = Field(None, description="The semantic object related to the verb, if there is one")

class SentenceSVO(BaseModel):
    """Represents all SVO relationships found in a single sentence."""
    svos: List[VerbSVO] = Field(..., description="One entry per verb found in the sentence")

# Get JSON schema to inject into prompt
svo_schema = SentenceSVO.model_json_schema()

# System prompt for SVO extraction
sysprompt_svo_zs = f'''
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

If there are multiple verbs or an object is separated into multiple observations as described above, make a new entry for each observation. Do not wrap in a code block. Do not return anything else.
'''

#### Define subfunctions ####

## Data read-in
def load_data(fp, column, local = False):
    if local == False:
        HOME = Path("/pfs/data6/home/hd/hd_hd/hd_gn354")

    df = read_tabular(fp)
    input = df['column']

    return input

# Checking for model presence and returning Boolean on whether <modelname> is available in the Ollama client
def model_check(client, modelname):
    try:
        client.show(modelname)
        return True
    except ollama.ResponseError:
        return False

# Main inference function
def inference(input, modelname, opts, thinking, logfile):
    '''
    Processes sentences through LLM and returns raw outs
    Workflow:
        - check for model availability
        - infer for each text
        - return cleaned raw messages
    '''

    client = ollama.Client()
    
    # Model availability:
    if not model_check(client, modelname):
        print(f"Model {modelname} cannot be found. Attempt to pull model.")
        client.pull(modelname)
        print("Model pulled successfully. Beginning inference.")

    # Set up counter and output list
    ctr = 0
    log_out = []
    out = []

    # Begin inference loop
    for text in input:
        messages = [
            {"role": "system", "content": sysprompt_svo_zs},
            {"role": "user", "content": text}
        ]

        response = client.chat(
            model = modelname,
            messages = messages,
            opts = opts,
            format = svo_schema,
            think = thinking
        )

        # Saving results and updating counter
        log_out.append(response)
        cleaned = response.message.content or "{}"
        out.append(cleaned)
        ctr += 1

    # Saving raw results and returning cleaned result
    with open(logfile, "w", encoding = "utf-8") as f:
        f.write(str(log_out))

    print(f"Successfully processed {ctr} texts. Raw outputs saved to {logfile}")

    return out

def validation(outputfile, pyd_schema):

    json_out = []

    for text in outputfile:
        try:
            parsed = json.loads(text)
            validated = pyd_schema(**parsed)
            json_out.append(validated.model_dump())
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Error: {e}")
            continue

    return json_out