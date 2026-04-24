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

ex_1 = '''[
    {
        "sentence": "One such tax would be a new carbon-based land tax for rural land.",
        "core_sents": [
            {
	            "type": "actor-issue",
	            "subject": "Scottish Greens",
	            "direction": "support",
	            "object": null,
	            "issue": "introducing carbon-based taxes"
            }
        ]
    }
]'''

ex_2 = '''[
    {
        "sentence": "Currently, renewable electricity sources cover just about half of Britain's electricity needs.",
        "core_sents": null
    }
]'''

ex_3 = '''[
    {
        "sentence": "Miliband also announced that he agreed with the EU's Ursula von der Leyen on a joint UK-EU climate investment fund to finance research projects in both areas.",
        "core_sents": [
            {
                "type": "actor-actor",
                "subject": "Labour",
                "direction": "support",
                "object": "European Union",
                "issue": "international climate investment funds"
            },
            {
                "type": "actor-actor",
                "subject": "European Union",
                "direction": "support",
                "object": "Labour",
                "issue": "international climate investment funds"
            },
            {
                "type": "actor-issue",
                "subject": "Labour",
                "direction": "support",
                "object": null,
                "issue": "international climate investment funds"
            },
            {
                "type": "actor-issue",
                "subject": "European Union",
                "direction": "support",
                "object": null,
                "issue": "international climate investment funds"
            }
        ]
    }
]
'''

ex_4 = '''
[
    {
        "sentence": "But the plan was also criticised by Green leader Zack Polanski, arguing that the government had not been bold enough in its action.",
        "core_sents": [
            {
                "type": "actor-actor",
                "subject": "Green Party",
                "direction": "opposition",
                "object": "Labour",
                "issue": "fighting climate change"
	        }
        ]
    }
]
'''

ex_5 = '''
[
    {
        "sentence": "Trump claimed Macron’s “wife treats him extremely badly” and even suggested that she hits him.",
        "core_sents": null
    }
]
'''

sysprompt_json = f'''
You are an expert coder with training and expertise in analysing political claims. You follow British politics to the level of an interested, engaged daily news reader, and know the most important politicians, parties, and issues in the United Kingdom in late 2025/early 2026. Your task is to analyze sentences from newspaper articles and extract the following variables according to the instructions provided:

1. Sentence: The sentence you are coding
2. Type: Identify whether the sentence describes an "actor-actor" or an "actor-issue" relationship.
3. Subject: Extract the name of the organization affiliated with the person making a claim or taking an action in the sentence.
4. Direction: Code the stance of the Subject towards the Object - one of "support", "opposition", or "ambiguous".
5. Object: Extract the name of the organization affiliated with the object of the sentence.
6. Issue reference: If the sentence references a political issue (either as the target of an actor-issue sentence or as a reference in an actor-actor sentence), describe it here.

Extract all variables present in the sentence. If there is no relationship expressed in a sentence, do not code the variables. If there is more than one relationship described in one sentence, extract the variables for each relationship separately.

## Detailed instructions

- Type: The type depends on the object of the sentence. If the subject is referring to another political or societal actor, such as a political party, a government, a protest movement, or a business interest, code "actor-actor". If the subject is referring to a political issue or position, code "actor-issue".
- Subject: The subject of the sentence should always be a political or societal actor, such as a party, politician, protest movement, business interest, etc. We are interested in the semantic subject expressing a stance, even if this differs from the grammatical subject. When there are multiple possible organisations for one actor, prioritise political parties when possible.
- Direction: Determine whether the subject supports or opposes the object or issue, and code accordingly. Only use "ambiguous" if the stance is truly unclear.
- Object: The object of a sentence should always be a political or societal actor. We are interested here in the semantic object, i.e. the target of the subject's stance, even if this differs from the grammatical object. When there are multiple possible organisations for one object, prioritise political parties when possible.
- Issue reference: If you have identified an actor-issue-relation, this is the issue the subject takes a stance towards. If you have identified an actor-actor-relation, this is any issue referenced to explain or justify the subject's stance towards the object.

## Input

You are given up to five sentences published in a British newspaper, one of which is marked with > <. Code only the marked sentence, but use the other sentences to provide context to the marked sentence.

## Output

Return the extracted variables according to the following JSON scheme:

{json.dumps(json_schema)}

## Examples

Input 1: The 165-page document calls for all bus services to be taken into public ownership and several new railway lines. It also suggests the introduction of 570 hours of free childcare each year for children between six months and two years old. Ross Greer, the co-leader of the Scottish Greens, said these policies would be funded by new taxes targeting the wealthy and large companies. > One such tax would be a new carbon-based land tax for rural land. < Other proposed taxes include health taxes on supermarkets selling tobacco and taxes on casinos and bookies.
Output 1:
{json.loads(ex_1)}

Input 2: Yesterday, Prime Minister Keir Starmer presented his government's plan for climate action in the UK. Key to Starmer's plan is a new scheme to subsidize wind and solar power in the UK. "We need to continue our transition to renewables, and our government wants to reach 75% renewable electricity in the UK by 2035", the PM said. > Currently, renewable electricity sources cover just about half of Britain's electricity needs. < He also announced an increase in taxes on heavily polluting industries to finance the scheme.
Output 2:
{json.loads(ex_2)}

Input 3: Speaking to the press at the COP30 conference in Belém last night, energy secretary Ed Miliband showed optimism about the future development of international climate policy. The Secretary claimed that the UK was excellently positioned to lead by example. "We are continuing to push for net-zero, and provide an example for other countries around the world", he said. > Miliband also announced that he agreed with the EU's Ursula von der Leyen on a joint UK-EU climate investment fund to finance research projects in both areas. < The details of the fund will be negotiated over the coming year.
Output 3:
{json.loads(ex_3)}

Input 4: The government's plan faced heavy criticism from multiple directions. Reform MPs decried it as "climate hysteria", and as overly punishing on average Brits. "Net zero is a dead end that will destroy the British economy for a green ideology", Reform leader Farage said. > But the plan was also criticised by Green leader Zack Polanski, arguing that the government had not been bold enough in its action. < "The government needs to get past the idea that they can have climate action without restricting energy consumption", he said.
Output 4:
{json.loads(ex_4)}

Input 5: Starmer and other European leaders have been repeatedly chastised and belittled by Trump and other prominent members of his administration. These have included Trump sharing a video from the sketch show SNL UK in which Starmer is portrayed as being scared of the US president. Others on the receiving end of Trump’s ire include the French president, Emmanual Macron. > Trump claimed Macron’s “wife treats him extremely badly” and even suggested that she hits him. < The Spanish prime minister, Pedro Sánchez, who has been outspoken in his disapproval of the wars in Iran and Gaza, has been one of Trump’s most vocal detractors.
Output 5:
{json.loads(ex_5)}
'''

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
        model = GPTLARGE,
        messages = messages,
        options = opts,
        format = json_schema
    )
    
    out.append(response)

transform_and_save(out, "output_baseline.json")