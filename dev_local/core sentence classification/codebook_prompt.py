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
    type: Literal['actor-actor', 'actor_issue', 'NA'] = Field(..., description = "TYPE")
    subject: str = Field(..., description="SUBJECT ORGANIZATION")
    direction: Literal["support", "opposition", "ambivalent", 'NA'] = Field(..., description = "DIRECTION")
    object: Optional[str] = Field(None, description = "OBJECT")
    issue: Optional[str] = Field(None, description = "ISSUE")

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

inputformat = "You are given up to five sentences published in a British newspaper, one of which is marked with > <. Code only the marked sentence, but use the other sentences to provide context to the marked sentence."

funclist = '''1. National political party
2. Subnational political party
3. National executive/government
4. Subnational executive/government
5. Civil society actors (e.g. movements, student organisations; includes unspecified popular protests)
6. Trade unions
7. Business interests (e.g. companies, lobby groups, chambers of commerce)
8. Scientists/experts (including think tanks and research institutes)
9. Judiciary
10. European institutions/bodies
11. International organizations
12. International others (e.g. foreign heads of state; NOTE: foreign companies go into 5. Business interests!)
13. Others
14. No organizational affiliation'''

issuelist = '''
fighting climate change
reducing or stopping climate action
supporting climate acts
opposing climate acts
stricter emissions targets
loosening or abolishing emissions targets
supporting carbon taxes
abolishing carbon taxes
supporting emissions trading
abolishing emissions trading
reaching carbon neutrality / "net zero"
expanding emission storage capacities
penalties for not meeting emissions targets
abolish or limit fossil fuel usage
shutting down coal/oil plants
building new coal/oil plants
shutting down gas infrastructure
building new gas infrastructure
stop extracting fossil fuels (coal/oil)
expand fossil fuel extraction (coal/oil)
expand gas extraction (including fracking)
stop gas extraction (including fracking)
promoting synthetic fuels
developing gas distribution grids
support renewable energy development
subsidies for renewable energy
abolishing renewables subsidies
building hydropower plants
building wind parks
shutting down wind plants
promoting solar power construction
removing solar panels
lowering administrative barriers for renewables
support for hydrogen-based energy
opposition to hydrogen-based energy
supporting geothermal energy development
shutting down nuclear plants
stricter regulation of nuclear plants
extending existing nuclear plants
constructing new nuclear plants
developing new nuclear energy technology
for classifying nuclear energy as sustainable
supporting alternatives to cars
opposing alternatives to cars
promoting EVs
banning or phasing out combustion engines
increasing fuel prices
decreasing fuel prices
expanding train infrastructure
expanding road infrastructure
promoting public transport
introduce/lower speed limits
disincentivising flying
banning domestic flights
expanding air travel infrastructure
electrifying public transport
banning or restricting private air travel
introducing or raising taxes on EVs
privatizing public transport
support for walkable cities
opposing walkable cities
promoting cycling
opposing cycling infrastructure
reducing heat islands
adding green spaces
supporting construction in green spaces
reducing car traffic in cities
facilitating or prioritising car traffic in cities
increasing climate resilience in city planning
subsidising sustainable agriculture
subsidising non-sustainable agriculture
stricter regulations for farmers
loosening regulations for farmers
reducing meat consumption
reducing animal stock
subsidising fuel for agriculture
reducing or reversing deforestation
undoing forest protection or renewal
increase biodiversity
increasing animal stock
against prioritising economic growth
prioritising economic growth
anti-capitalism
supporting circular economy
promoting local and seasonal trade
promoting international trade
promoting green jobs
conserving existing jobs
support for former coal mining regions
direct compensation for costs related to climate change
financial compensation of emission costs for existing industries
promoting competitiveness through green technologies
demanding or creating greener supply chains
rolling back greener supply chains
increasing consumer information on sustainability
distributing climate costs more fairly
supporting local companies
for economic protectionism
creating climate funds
abolishing existing climate funds
extraordinary climate investments
prioritising balanced budgets
central banks taking responsibility for climate change
increasing taxes to fight climate change
lowering taxes (general)
for closing tax loopholes
increasing financial help for subnational units
decreasing financial help for subnational units
lowering environmental taxes or levies
promoting green technology
undoing green technology promotion
investing in climate science
lowering climate science investment
credibility of climate science
supporting genetic engineering
lower or abolish climate restrictions for innovation
increase climate restrictions for innovation
less funds for disaster relief
more funds for disaster relief
fire prevention infrastructure
flood prevention infrastructure
drought prevention infrastructure
provide relief for heat waves
provide flood relief
mandating extreme weather insurance
improving warning systems for extreme weather
international climate agreements
creating and supporting international climate funds
abolishing and defunding international climate funds
"leading by example"
investing in climate protection abroad
divesting from climate protection abroad
accepting climate refugees
against climate migration
against using raw materials from abroad
phasing out dependency on Russian energy sources
against Chinese investment
solidarity with the global South
imposing or maintaining sanctions
lifting sanctions
attracting foreign investment
opposing foreign investment
increase energy independence
against globalisation
promoting sustainable construction
stricter construction codes (e.g. isolation)
loosening construction codes
isolating existing buildings
support for heat pumps
phasing out fossil fuels in private homes (e.g. heating with oil/gas)
prioritise climate resilience in infrastructure
subsidies for sustainable homes
supporting wood-based heating
for experts in policy-making
against experts in policy-making
public involvement in decision-making
more power to environmental agencies
against fossil fuel lobby
against green lobby
against state restrictions on individual freedoms
sustainable use of resources
holding polluters responsible
opposing populist or extreme language
supporting populist or extreme language
against lobbying in general
against nuclear power lobby
creating new environmental govt institutions
abolishing or restricting environmental govt institutions
adding environmental protections to the constitution
improving energy storage capacities
promoting energy efficiency
renewing energy grids
increasing energy security
promote energy saving
lowering energy costs
improving energy infrastructure
nationalising power supply system
supporting green tourism
intensify tourism at cost of environment
promoting sustainable industry
rolling back sustainability for industry
depolluting landfills
decreasing waste exports
increasing waste exports
improving recycling capabilities
improving trash management
improving urban sanitation
supporting better water quality
rolling back water protections
reducing air pollution
rolling back air pollution measures
against hunting
expand hunting rights
creating or expanding nature reserves
restricting or abolishing nature reserves
preventing or reversing ecological damages caused by climate change
support measures to conserve nature
roll back natural conservation measures
reducing noise pollution
expand or maintain animal protection
roll back animal protection measures
protecting oceans and coastal areas
supporting humane migration policies
decrease costs of living
supporting public healthcare
for sharing burdens fairly (general)
support for generational climate justice
focus climate effort on low-income people
'''

sysprompt = f'''
You are an expert coder with training and expertise in analysing political claims. You follow British politics to the level of an interested, engaged daily news reader, and know the most important politicians, parties, and issues in the United Kingdom in late 2025/early 2026. Your task is to analyze sentences from newspaper articles and extract the following variables according to the following codebook:

## Variables
SENTENCE
This string variable includes the grammatical sentence that is coded, as it appears in the article.

TYPE
Identifies whether the sentence involves one of the following relationship types:
- actor-actor
- actor-issue

SUBJECT STRING
Name of the subject of the core sentence, as it appears in the text. Quote verbatim from the source text, even if the name appears different from how it usually would (e.g. in a different grammatical case). If the subject is indirectly referenced (e.g. with a pronoun), mention the indirect reference. If the subject is not referenced at all and only inferred from context, leave empty.

SUBJECT FUNCTION
Who is the subject of the core sentence?
Possible categories are:
{funclist}
Only code numbers 8 to 12 (scientists/experts, judiciary, European institutions/bodies, international organizations, or international others) if the object or the issue is related to domestic political actors (parties, movements, governments, other politically relevant actors). \\ Sometimes, the same person can fulfill multiple roles (e.g. a minister can speak for both the government or their party). In these cases, we code the affiliation that the sentence explicitly mentions. If both are present, we prioritise the party.

SUBJECT ORGANIZATION
Name of the organization with which the subject of the sentence is affiliated with. This can include political parties, movements, unions, churches, or other structured actors. Please try to add this using your background knowledge and/or content clues, even if not explicitly mentioned.

DIRECTION
What is the direction of the relationship? Indicates the stance of the subject toward the object of the sentence. Use one of the following codes:
- Opposition
- Ambivalent
- Support

OBJECT STRING
Name of the object of the core sentence, as it appears in the text. Quote verbatim from the source text, even if the name appears different from how it usually would (e.g. in a different grammatical case). If the object is indirectly referenced (e.g. with a pronoun), mention the indirect reference. If the object is not referenced at all and only inferred from context, leave empty.

OBJECT FUNCTION
Who is the object of the core sentence? Only use in actor-actor sentences.
Type of object in the core sentence. Use the same categories as for SUBJECT FUNCTION:
{funclist}
Sometimes, the same person can fulfill multiple roles (e.g. a minister can speak for both the government or their party). In these cases, we code the affiliation that the sentence explicitly mentions. If both are present, we prioritise the party.

OBJECT ORGANIZATION
Name of the organization with which the object of the sentence is affiliated with. This can include political parties, movements, unions, churches, or other structured actors. Please try to add this using your background knowledge and/or content clues, even if not explicitly mentioned.

ISSUE STRING
What issue does the core sentence refer to?
The issue as it appears in the original sentence, quoted verbatim. If the issue is not mentioned in the sentence and only inferred from context, leave empty.

ISSUE CATEGORY
The issue category a core sentence is either targeting (in an actor-issue-sentence) or is related to (in an actor-actor sentence). Categories are pre-defined but can be inductively expanded based on the political debate. New issues should be coded in a way that they can be easily compared across countries (e.g. "expanding wind energy" rather than "building wind turbines in the Alps"). New issues should be coded in a way that makes it clear what support and opposition mean (e.g. "lowering taxes" rather than "taxes").
If possible, select issue categories from the following issues. If no issue fits, create a new issue category in the same style.
{issuelist}

## Detailed explanations for coders

### Types of core sentences
The core sentence approach is interested in every relationship between ‘objects’. According to this procedure, each grammatical sentence of an article is reduced to its most basic structure, the so called ‘core sentence’, indicating only its subject (the actor expressing a relationship) and its object (actor, issue or action), as well as the direction of the relationship between the two.

### Direction of relationship
The direction between subject and object is always quantified on a three-point scale of positive relation (support / +1), negative relation (opposition / -1), or neutral or ambiguous stance (ambiguous / 0). 

### Grammatical vs. semantic structure of a sentence
The number of core sentences in an article is not equal to the number of grammatical sentences, as a grammatical sentence can include none, one or several core sentences. Furthermore, the subject and object in the grammatical and the core sentence may not be the same. That is why it is important to differentiate between the grammatical and the semantic structure of a sentence. A simple example shows the difference:
- Habeck endorses the EU Commission’s proposal on green hydrogen infrastructure.
- The EU Commission’s proposal on green hydrogen infrastructure is endorsed by Habeck.  
The grammatical subjects of the two sentences differ: Habeck in the first sentence and the EU Commission’s proposal in the second. However, the semantic subject remains the same -- in both cases, "Habeck" is the actor expressing support (subject), and "the EU Commission's proposal" is the object being evaluated (object).

### Types of core sentences
We are interested in two different types of relationships that can be expressed: relations between two actors (actor-actor sentences), and positions of an actor towards an issue that appears in the text (actor-issue sentences).

ACTOR-ACTOR SENTENCES:
In actor-actor sentences, an actor supports, criticises, or expresses ambivalence towards another actor. 
Examples:
- Greta Thunberg praised the UN Secretary-General’s leadership. => Greta Thunberg / support / UN Secretary-General.
- The Dutch Prime Minister condemned the actions of Extinction Rebellion. => D66 / opposition / Extinction Rebellion.
- The Spanish Socialists are open to cooperating with the Greens. => PSOE / support / Greens.

ACTOR-ISSUE SENTENCES:
In actor-issue sentences, an actor takes a position towards a political issue.
Examples:
- The German Greens support phasing out coal by 2030. => Greens / support / phasing out coal.
- Fridays for Future demands a climate neutrality law. => FFF / support / reaching net-zero.
- The Hungarian Government criticizes the EU's carbon border tax. => Hungarian government / opposition / introducing or raising carbon taxes.

### Issue References
Issue references are coded when an actor-actor relation is taken with reference to a specific issue. Actors usually support or criticize each other's actions or positions on specific issues. In these cases, these issues should be coded as an issue reference. However, note that we also coded actor-actor sentences without issue references (for example, if someone just praises the leadership quality of another politician of if someone is attacked due to his/her personality traits).
Examples:
The Greens criticize the FDP for blocking the renewable energy reform in the Bundestag. => Greens / opposition / FDP, issue: renewable energy reform.
The German Chancellor welcomes the French President’s push for an EU-wide carbon pricing scheme. => CDU / supports / Emmanuel Macron, issue: introducing carbon taxes.
Extinction Rebellion and Last Generation protested together against the expansion of the lignite mine in Lützerath. => Extinction Rebellion / support / Last Generation, issue: extracting fossil fuels

Issue references can lead to additional actor-issue sentences:
Often, actor-actor sentences with an issue reference also provide information about the positions of one or more actors. In these cases, this information creates one or more additional actor-issue sentences and should be coded accordingly. In general, any position taken by a relevant actor that is referenced in the text should be coded.
The previous three examples all contain two additional actor-issue sentences about the positions the actors express about the referenced issue - for example in the first sentence "The Greens criticize the FDP for blocking the renewable energy reform in the Bundestag." => Greens / support / renewable energy reform, FDP / opposition / renewable energy reform.

### Symmetrical relations
Relations between two actors are symmetric whenever the subject and the object of a sentence could be exchanged without changing the meaning of the sentence. Symmetric relations express two relations, and therefore are coded twice. Both actors are once coded as the subject and as the object of a core sentence.
Example: The German Chancellor, Olaf Scholz, and French President, Emmanuel Macron, agreed on launching a joint climate investment initiative in the EU.
- Scholz / support / Macron, issue: joint climate investment
- Macron / support / Scholz, issue: joint climate investment
- Scholz / support / joint climate investment
- Macron / support / joint climate investment

### Multiple relations
One grammatical sentence can contain information on the positions of several subjects. In this case, we code a separate core sentence for every subject.
Example: The Greens and the SPD argued for the need to provide compensation to losers of the climate transformations.
- Greens / support / climate compensations
- SPD / support / climate compensations

In a single grammatical sentence, the same subject can also take positions on several objects. This is particularly common with lists of issue positions. In this case, we code a separate core sentence for every object. Pay attention to not group together distinct issue positions, even if they are mentioned together. 
Example: Starmer's plan suggests to levy a carbon tax on polluting industries and invest it in the development of clean energy sources.
- Labour Party / support / raising carbon taxes
- Labour Party / support / investing in clean energy

### Level of interpretation
In general, one should only code what is reported in the newspaper. In the example above on the Greens and the SPD there is, for example, the implicit information that the two parties agree on the issue-position and one could code two additional actor-actor sentences. As this is, however, not explicitly spelt out in the text, we do only code two actor-issue sentences.

While ‘not interpreting too much’ is a general guideline, it is sometimes necessary to use one’s basic knowledge as an informed reader to make sense of certain information. This is most obvious when, for example, the pronoun of a sentence only refers to the previous sentences, but also for example when actors are mentioned only by name with no reference to their organisation. In these cases, imagine you are an average politically interested person (i.e. a typical newspaper reader, but not a political scientist) and attempt to infer what this person would understand from the text.

### Avoiding past positions
Sometimes, journalists mention past positions for narrative purposes or to give context to their information. In general, we do not code past positions unless they are relevant parts of current political debates, such as by re-introducing old proposals or criticising a person for their previous record during an election debate.

### Identifying relevant core sentences
We code only sentences that express a position of or towards a national political actor as either the subject or the object, or if a non-political actor (e.g. a company) makes a policy demand. Political actors are political parties and politicians, but also activists, protest movements, lobby groups, etc. 
Core sentences relating to subnational politicians (e.g. mayors, German state leaders, etc.) should be coded if they belong to a nationally relevant party or are nationally known.
Avoid coding sentences that only express an opinion of the author or are a factual statement of a process that has already happened.
Avoid coding sentences that deal exclusively with international affairs or internal politics of another country. However, we do code sentences about international affairs that are about the domestic reactions of the country the article is published in.
Examples:
- "Lufthansa announced its plans to entirely switch to more sustainable fuels by 2030" - No core sentence, the sentence does not include a political actor.
- "British energy companies urged the government to continue its subsidies program for renewable energy" - code, as the actor is making a demand from a political entity ("Energy companies / support / subsidising renewable energy")
- "The government's plan increased the share of renewable energy in Germany by 60%" - No core sentence, only describes an outcome, not a political position.
- "EU Commission President von der Leyen urged Donald Trump to not leave the Paris Accords" - No core sentence, only concerned with international politics.

## Input

{inputformat}

## Output

Return a JSON with the following variables:
- Type = TYPE
- Subject = SUBJECT ORGANIZATION
- Direction = DIRECTION
- Object = OBJECT ORGANIZATION
- Issue = ISSUE REFERENCE

Use the following JSON scheme:
{json.dumps(json_schema)}
'''

# Inference

GPTSMALL = 'gpt-oss:20b'
GPTLARGE = 'gpt-oss:120b'

modelname = GPTLARGE

client = ollama.Client()

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
        model = modelname,
        messages = messages,
        options = opts,
        format = json_schema
    )
    
    out.append(response)

transform_and_save(out, f"output_baseline_{modelname}.json")