import pandas as pd
import numpy as np
import ollama
import re
import json
import argparse

from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ValidationError

# Import custom IO module
from src.io import read_tabular

# Define Home for Cluster
HOME = Path("/pfs/data6/home/hd/hd_hd/hd_gn354/projects/core_sentence_pipe")

# Define Pydantic models for SVO extraction
class VerbSVO(BaseModel):
    """Represents a single Subject-Verb-Object relationship."""
    sentence: str = Field(..., description="The full grammatical sentence you extracted the SVO from")
    subject: str = Field(..., description="The semantic subject related to the verb")
    verb: str = Field(..., description="The verb of the sentence or partial sentence")
    object: Optional[str] = Field(None, description="The semantic object related to the verb, if there is one")

class SentenceSVO(BaseModel):
    """Represents all SVO relationships found in a single sentence."""
    svos: List[VerbSVO] = Field(..., description="One entry per verb found in the sentence")

# Get JSON schema for validation
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

If there are multiple verbs or an object is separated into multiple observations as described above, make a new entry for each observation. Do not wrap in a code block. Do not return anything else.'''

def data_load(path_string, varname):
    fp = Path(path_string)
    df = read_tabular(fp)
    input = df[varname]
    return input
  
def model_exists(client, modelname):
    """Check if a model exists in the Ollama client."""
    try:
        client.show(modelname)
        return True
    except ollama.ResponseError:
        return False

def process_sentences(input_data, modelname, output_file, error_file, opts, thinking=False):
    """
    Process sentences through the LLM and extract SVO relationships.
    
    Args:
        input_data: List of sentences to process
        modelname: Name of the LLM model to use
        output_file: Path to save validated JSON output
        log_file: Path to save raw responses
        opts: Model options dictionary
    
    Returns:
        Tuple of (success_count, error_count, total_count)
    """
    client = ollama.Client()
    
    if not model_exists(client, modelname):
        print(f"Model {modelname} not found. Pulling model.")
        client.pull(modelname)
        print("Model pulled successfully.")
    
    ctr = 0
    error_ctr = 0
    success_ctr = 0
    out = []
    
    for text in input_data:
        messages = [
            {"role": "system", "content": sysprompt_svo_zs},
            {"role": "user", "content": text}
        ]
        
        response = client.chat(
            model=modelname,
            messages=messages,
            options=opts,
            format=svo_schema,
            think=thinking
        )
        
        cleaned = response.message.content or "{}"
        ctr += 1
        out.append(cleaned)
    
    # Validate and save JSON output
    json_out = []
    error_out = []
    
    for text in out:
        try:
            parsed = json.loads(text)
            validated = SentenceSVO(**parsed)
            json_out.append(validated.model_dump())
            success_ctr += 1
        except (ValidationError, json.JSONDecodeError) as e:
            errortxt = f"Error: {e} on following output: {error_out}"
            error_out.append(errortxt)
            error_ctr += 1
            continue
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    with open(error_file, "w") as f:
        f.write(str(error_out))
    
    return success_ctr, error_ctr, ctr

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract SVO relationships from sentences via an Ollama model."
    )
    parser.add_argument(
        "-m", "--model",
        default="gemma4:31b",
        help="Ollama model name (default: %(default)s)"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("../../data/test_semantic_extraction.csv"),
        help="Input CSV file with a 'sentence' column (default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON file. Defaults to svo_<model-tag>.json next to the input."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Raw-response log file. Defaults to <output>_log.txt"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-sentences", type=int, default=None,
        help="Process only the first N sentences (for quick test runs)."
    )
    parser.add_argument("--thinking", type=bool, default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    model_tag = HOME / args.model.replace(":", "_").replace("/", "_")
    output_file = HOME / args.output or Path(f"svo_{model_tag}.json")
    log_file = HOME / args.log_file or Path(f"{output_file.stem}_log.txt")

    opts = {"seed": args.seed, "temperature": args.temperature}

    input_data = read_tabular(HOME / args.input)
    if args.max_sentences:
        input_data = input_data[: args.max_sentences]

    success, errors, total = process_sentences(
        input_data, args.model, output_file, log_file, opts, args.thinking
    )

    print(f"\nProcessing complete!")
    print(f"Total: {total} | Success: {success} | Errors: {errors}")
    print(f"Results: {output_file} | Error log: {log_file}")

if __name__ == "__main__":
    main()