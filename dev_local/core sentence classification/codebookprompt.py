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
input = df_uk['contexted']#[0:10]

## Pydantic setup

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

response_scheme = CSResponse.model_json_schema()

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

## Prompts

cs_instructions = [
    {
      "Category": "Simple relations",
      "Definition": "Code one core sentence if a grammatical sentence contains one relation between a subject and a target.",
      "Clarification": "Whenever a subject takes a stance towards a target, or takes an action against a target, there is a relation between them. The subject is always the actor that is taking the action or stance, even if they are not the grammatical subject of the sentence.",
      "Negative Clarification": "The subject needs to actively make a claim towards the target. If the claim is not a direct relation between the two, but only an interpretation or background information given by the writer, do not code.",
      "Positive Example": "\"Chancellor Merz continues to receive support from the SPD\". There is a stance taken by the SPD towards the Chancellor. Code one core sentence.",
      "Negative Example": "\"The government has not created more jobs in green technology than there were when they took office\". This is just background information given by a journalist. The sentence does not make clear whether the government has attempted to create green jobs or supports this goal. Do not code a core sentence."
    },
    {
      "Category": "Multiple relations",
      "Definition": "Code multiple core sentences if one grammatical sentence expresses more than one relation between independent subjects and/or targets.",
      "Clarification": "Usually, this is expressed either by multiple subclauses, or by a list of actors and/or targets that are referenced in the same action. Carefully analyze all subclauses for further relations.",
      "Negative Clarification": "Do not code multiple relations if the actors/targets mentioned are all part of the same organisation or issue category.",
      "Positive Example": "\"Tories and Reform have thrown heavy criticism at the Energy Secretary\". \"Tories\" and \"Reform\" are two independent actors. Code two core sentences, one for the relation between \"Tories\" and \"the Energy Secretary\" and one for the relation between \"Reform\" and \"the Energy Secretary\".",
      "Negative Example": "\"Macron wants to subsidise the creation of wind farms and construction of solar panels\". \"Wind farms\" and \"solar panels\" form a group of targets that both fall under RENEWABLES. Code as one core sentence for the relation between \"Macron\" and \"creation of wind farms and construction of solar panels\"."
    },
    {
      "Category": "Inferred relations",
      "Definition": "Code multiple core sentences if one grammatical sentence expresses a (dis)agreement between actors as well as their respective positions.",
      "Clarification": "When two actors (dis)agree with each other on an issue, they frequently also express issue positions implicitly. Code the actor-actor relation between the actors, as well as the issue positions communicated by the sentence.",
      "Negative Clarification": "Do not code issue positions that are not referenced in the sentence based on background knowledge. If an issue is referenced, but no stances are provided, only code the relation between actors.",
      "Positive Example": "\"Fridays for Future criticises the government for their reluctance to take radical climate action\". The sentence mentions a relation between \"Fridays for Future\" and \"the government\", but also indicates issue positions (Fridays for Future are in favour of radical climate action, the government is against radical climate action). Code three core sentences.",
      "Negative Example": "\"Greens and CDU clash over combustion engine exit\". Although an issue is given, the sentence does not provide information on where these parties stand on the issue. Code only one core sentence for the relation between Greens and CDU."
    },
    {
      "Category": "Symmetric relations",
      "Definition": "Code two symmetric core sentences if the subject and target of a sentence take mutual action without a clear leader.",
      "Clarification": "Symmetric core sentences occur when the subject and object of a grammatical sentence can be interchanged without changing the meaning of the sentence. In this case, code each actor once as subject and once as target of a core sentence.",
      "Negative Clarification": "If there is a clear indication that one actor initiated the action, interchanging the subject and object of the sentence changes the meaning. In this case, only code one core sentence for the relation. For symmetric actor-actor relations, code the direction as support when the actors jointly cooperate, coordinate, agree, form an alliance, launch a shared initiative, or otherwise act together toward a shared goal. Code the direction as opposition when the actors mutually clash, compete, dispute, blame each other, or are described as being in conflict without a clear initiator. Use ambiguous only when the sentence establishes a symmetric relation but does not make the direction clear.",
      "Positive Example": "\"Macron and Meloni announced a joint initiative to coordinate climate action in the Alps\". The role of \"Macron\" and \"Meloni\" in this sentence is interchangeable. Code two core sentences for the relation of Macron towards Meloni and of Meloni towards Macron. Direction → support for both. \"Greens and CDU clashed over the combustion engine exit\" → code two actor-actor core sentences: Greens oppose CDU, and CDU opposes Greens. Direction → opposition for both.",
      "Negative Example": "\"Scholz wants to join Macron's European climate initiative\". Here, \"Scholz\" and \"Macron\" have different roles, and interchanging them would change the meaning of the sentence. Code only one core sentence between \"Scholz\" and \"Macron\"."
    }
  ]

types = [
    {
        "Category": "actor-actor",
        "Definition": "The target of a core sentence is another actor.",
        "Clarification": "Actors are organised political, labour, or business interests. To be an object of a core sentence, an actor does not have to be part of national politics - it can also be a company, an expert (group), or a foreign politician.",
        "Negative Clarification": "Multiple actors being mentioned does not always imply an actor-actor relation. In order for a core sentence to be an actor-actor sentence, one of the actors needs to take a stance on another actor. Multiple people acting on behalf of the same organisation do not constitute an actor-actor relation.",
        "Positive Example:": "\"Le Pen criticized Macron's lack of leadership\". Both the people mentioned are national political actors, and there is one actor (Le Pen) criticizing another (Macron).",
        "Negative Example": "\"Starmer and Miliband propose a new plan to relieve Brits of high energy costs\". The people mentioned represent the same group (the Labour Party) and should be considered as one actor. Code as an actor-issue sentence."
    },
    {
        "Category": "actor-issue",
        "Definition": "The target of a core sentence is a political issue.",
        "Clarification": "Often, actors just take a stance on a specific political issue in the form of a demand, a claim, or by supporting or opposing a proposed plan. In these cases, code an actor-issue sentence.",
        "Negative Clarification": "Generic references to a person's political qualities, such as criticising someone's honesty, do not constitute actor-issue relations even when an issue is mentioned or known to you from background information. Only use this category if the sentence explicitly states a position (including ambivalence) towards an issue.",
        "Positive Example": "\"Giorgia Meloni announces her plans to replace Russian gas with American LNG\". The statement shows a clear stance by an actor (Meloni) towards an issue (replacing Russian gas with American LNG).",
        "Negative Example": "\"Greens claim Merz was dishonest about his position on the debt brake.\" While an issue is mentioned, there is no position-taking being mentioned from either side - the claim is about dishonesty. Code as an actor-actor relation."
    }
]

corevars = [
    {
      "Variable": "subject",
      "Definition": "The subject of a core sentence is an actor who makes a political claim in the grammatical sentence.",
      "Clarification": "A subject needs to be a nationally relevant political actor, such as a party, movement, or a civil society actor. Businesses, trade unions, experts, etc. can be actors too if they formulate national political claims. The subject is not necessarily equal to the grammatical subject.",
      "Negative Clarification": "We do not code subjects where they are foreign actors. For example, Donald Trump pulling out of the Paris Agreement would not be coded as we are not interested in US politics. This also applies to European Union-level institutions, parties, or individual politicians, actors that we only code when they express a position in relation to domestic actors or issues. We do not code non-political actors unless they are making political demands or comment on the actions of political actors.",
      "Positive Example": "\"Friedrich Merz wants to reduce subsidies for renewable energies\" → \"Friedrich Merz\"",
      "Negative Example": "\"Volkswagen announced their intentions to phase out production of combustion engine cars by 2040.\" The actor \"Volkswagen\" is not explicitly linking its plans to national politics; therefore, it is not making a political claim. We do not code it as a subject."
    },
    {
      "Variable": "subject_organisation",
      "Definition": "The organisation (political party, institution, movement, etc.) the actor making a claim is part of or related to.",
      "Clarification": "Organisations do not need to be actively mentioned to be coded. Use context and background knowledge to determine the organisation. Many actors are part of one or more organisations (political parties, institutions, movements etc.). In this case, prioritise parties or movements, or use the organisation mentioned in the text.",
      "Negative Clarification": "Do not code past organisational affiliations. For politicians, use party affiliations rather than institutional affiliations wherever possible.",
      "Positive Example": "\"Le Pen called Macron's climate plan 'the destruction of the French economy'\" → \"Rassemblement National\"",
      "Negative Example": "\"Chancellor Merz announced his plans to propose a climate fund at the upcoming EU summit\". As we have an individual with a clear party affiliation, the correct code here is not \"German Government\", but \"CDU/CSU\"."
    },
    {
      "Variable": "direction",
      "Definition": "The position (coded as \"support\", \"opposition\", or \"ambiguous\") that the subject takes towards the issue or object of the core sentence.",
      "Clarification": "In an actor-actor sentence, the direction refers to how the subject positions itself towards the object. In an actor-issue sentence, the direction refers to how the subject positions itself towards the issue referenced in the sentence. Try to assign a clear direction (i.e. \"support\" or \"opposition\") whenever possible.",
      "Negative Clarification": "In an actor-actor sentence, disregard the issue when coding direction. Do not use background knowledge for this variable - only code the direction as it appears in the specific sentence in the text.",
      "Positive Example": "\"Extinction Rebellion protested against the government's plan\" → \"opposition\"",
      "Negative Example": "\"The Greens voted against the proposed law to subsidise renewables\" → The text mentions opposition of the Greens (\"voted against\"). Code as the Greens opposing renewable subsidies, even if you know they generally support subsidies for renewables."
    },
    {
      "Variable": "object",
      "Definition": "The object of a core sentence is an organised actor (party, movement, business, etc.) that is the target of a political claim.",
      "Clarification": "Objects do not necessarily have to be nationally relevant actors. Any organisation or actor that is talked about by a relevant subject is a valid object to be coded, even if it is non-political (e.g. businesses) or from another country.",
      "Negative Clarification": "Do not code an object in an actor-issue sentence. Objects can only be actors (i.e. people or organisations). Do not code abstract or vague actors (e.g. \"politicians\", \"the industry\").",
      "Positive Example": "\"Fridays for Future's actions were criticised by PM Starmer in his parliamentary speech\" → \"Fridays for Future\"",
      "Negative Example": "\"Meloni also emphasised the importance of a secure gas supply for the Italian industry\". The target of this sentence is \"a secure gas supply\", which is an issue rather than an actor. Do not code an object. \"the Italian industry\" is too vague to be considered an object of a core sentence."
    },
    {
      "Variable": "object_organisation",
      "Definition": "The organisation (political party, institution, movement, etc.) the target of a claim is part of or related to.",
      "Clarification": "Organisations do not need to be actively mentioned to be coded. Use context and background knowledge to determine the organisation. Many actors are part of one or more organisations (political parties, institutions, movements etc.). In this case, prioritise parties or movements, or use the organisation mentioned in the text.",
      "Negative Clarification": "Do not code past organisational affiliations. For politicians, use party affiliations rather than institutional affiliations wherever possible.",
      "Positive Example": "\"Extinction Rebellion blockaded and attacked a Shell office building\" → Shell. Even though Shell is not a political actor, it is the target of a political action.",
      "Negative Example": "\"Nigel Farage demands an end to the government's 'climate mania'\" → \"UK Government\". Use background knowledge to prioritise party affiliation, rather than coding institutions, where possible - the correct code would be \"Labour Party\"."
    },
    {
      "Variable": "issue",
      "Definition": "An issue that is being referenced in a core sentence, either directly as the target of the sentence or in relation to the target.",
      "Clarification": "In actor-issue sentences, the issue is the target of the core sentence, i.e. the thing that a claim is being made about. In actor-actor sentences, an issue can also be referenced as a justification or a reason for the statement. Code both these instances as issues. Issues always need to be coded as a position or action where \"supports\" or \"opposes\" has a clear meaning.",
      "Negative Clarification": "Do not code issues that are simply mentioned without being connected to the claim. Do not code instances where an actor is simply mentioned in conjunction with an issue, but no claim is being made, such as a simple description of a phenomenon.",
      "Positive Example": "\"The Romanian Prime Minister announced his plan to create a fund for relief for flood victims\" → \"relieve flood victims\". This is the target of the core sentence. \"Extinction Rebellion protested the government's decision to allow an extension of coal mining\" → \"expanding coal mining\". The target of the sentence is \"the government\", but coal mining is referenced as the reason for opposing the government.",
      "Negative Example": "\"Meloni says she is preparing a relief package for high fuel costs.\" → \"fuel costs\". It is unclear what this means. Use a clearer directional label such as \"relief for fuel costs\" or \"lowering fuel costs\"."
    }
]

issue_cats = [
    {
      "Category": "CLIMATE CHANGE (general)",
      "Definition": "General positions related to climate change without a reference to any specific sub-issue",
      "Clarification": "Use for generic positions that avoid specific issue references, as well as for cross-cutting statements about the climate crisis as a whole.",
      "Negative Clarification": "If climate change is invoked to justify action in a more specific area, use the more specific issue area instead. For references to \"net zero\" use EMISSIONS.",
      "Positive Example": "\"Solving the climate crisis needs to be the government's top priority\". The claim targets climate change at large, without any specific issue reference.",
      "Negative Example": "\"Reducing the role cars play in our society is a key piece of addressing climate change\". The specific reference to cars makes TRANSPORT a more appropriate code."
    },
    {
      "Category": "EMISSIONS",
      "Definition": "References to greenhouse gas emissions that are not tied to a specific source",
      "Clarification": "Used primarily for general carbon and greenhouse gas emission targets such as emission targets or policies that are source-neutral such as carbon taxes. Includes references to \"Net-zero\".",
      "Negative Clarification": "Does not include specific references to fossil fuel use such as fossil fuel based electricity or transportation.",
      "Positive Example": "\"Our plan includes various measures to make Germany climate neutral by 2050\". The reference to climate neutrality indicates reducing greenhouse gas emissions, but no specific source is mentioned.",
      "Negative Example": "\"The energy sector needs to move away from fossil fuels as soon as possible\", the Green party leader said. The specific mention of the energy sector means that FOSSIL FUELS is the appropriate code."
    },
    {
      "Category": "ENERGY USE",
      "Definition": "References to the usage of energy and the energy infrastructure, independent of source.",
      "Clarification": "Use for references to energy that are independent of specific energy sources. Energy includes both electrical energy and other forms of energy such as oil heating or gas use for cooking. Also code ENERGY USE for references to energy infrastructure such as grids.",
      "Negative Clarification": "For references to specific energy sources, use FOSSIL FUELS, RENEWABLES, or NUCLEAR ENERGY. For references to construction measures to reduce or change energy use in homes, use CONSTRUCTION.",
      "Positive Example": "\"Government wants to invest into modernising the energy grid\". This is a reference to energy infrastructure broadly, without being linked to a specific source or use.",
      "Negative Example": "\"The Chancellor emphasised that there is no way around reducing the use of fossil energy in the long term\". This specifies a source of energy the reference is about. Therefore use FOSSIL FUELS instead."
    },
    {
      "Category": "FOSSIL FUELS",
      "Definition": "References to the extraction and use of fossil fuels.",
      "Clarification": "Fossil fuels refers to the use of coal, oil, and natural gas. This category includes both the extraction of fossil fuels and their use to provide energy more broadly, including infrastructure for their use.",
      "Negative Clarification": "Do not code references to using fossil fuels in transportation or heating here. Code such references as TRANSPORT or CONSTRUCTION respectively.",
      "Positive Example": "\"The protestors demanded a stop to the plans to expand coal mining in the Hambacher Forst\". Coal mining is a typical form of extracting fossil fuels.",
      "Negative Example": "\"The CDU group rejects a forced transition away from oil heating.\" Heating is a specific aspect of another sector with its own code. Code as CONSTRUCTION."
    },
    {
      "Category": "RENEWABLES",
      "Definition": "References to expanding or opposing the development of renewable energy.",
      "Clarification": "Expanding the development of renewables refers to any sort of subsidies, direct investment, constructing of renewable plants, or lowering administrative barriers to construct renewable energy plants. This category includes both large projects like wind parks and small-scale projects like solar cells on houses.",
      "Negative Clarification": "Do not include references to nuclear energy even when it is framed as renewable or \"clean\" energy. Do not include references to electrifying other sectors such as transport, heating, etc.",
      "Positive Example": "\"The Romanian government revealed plans to construct what would be Romania's largest hydroelectric dam.\" A plan to construct a hydropower plan is a promotion of renewable energy.",
      "Negative Example": "\"Macron urged France's transition to clean energy and emphasised the important role of nuclear plants in the process\". References to \"clean\" energy are not the same as references to \"renewable energy\". The main emphasis here is on nuclear energy rather than renewables, therefore code as NUCLEAR ENERGY instead."
    },
    {
      "Category": "NUCLEAR ENERGY",
      "Definition": "References to using, developing, or abolishing nuclear power.",
      "Clarification": "Code any reference to nuclear energy generation under this category, regardless of whether it is related to nuclear as a potential solution to climate change or relating to other topics related to nuclear energy such as security risks.",
      "Negative Clarification": "This category is exclusively about civil uses of nuclear technology for power generation. Do not code any references to other nuclear technology, such as nuclear weapons or medical uses of radiotherapy.",
      "Positive Example": "\"French energy minister announces plans to expand nuclear program\". The mention of the energy minister makes clear this is about nuclear as a power source.",
      "Negative Example": "\"Merz criticised the Iranian nuclear program as a 'great danger to global security'\". The context of global security and the Iranian program makes it clear this is most likely about military uses, rather than power generation. Do not code at all."
    },
    {
      "Category": "TRANSPORT",
      "Definition": "References to addressing climate change through any sort of transportation and traffic policy, or opposition to such policies.",
      "Clarification": "Transportation policy includes any references to cars, roads, trains, airports etc. This category includes both personal mobility and shipping. Code references to policies related to the price of fuel with this category.",
      "Negative Clarification": "Code references to the built environment of cities (e.g. cycling lanes) with CITY PLANNING. Code broader references to oil prices as FOSSIL FUELS unless they are explicitly linked to fuel prices.",
      "Positive Example": "\"Meloni plans to introduce a national cap on fuel prices to help rural Italians relying on their cars through the crisis\". Fuel prices are a part of transportation policy.",
      "Negative Example": "\"The mayor of Paris has invested heavily in bike lanes and other alternatives to cars\". The focus on the built environment of the city (bike lanes) means this should be coded as CITY PLANNING."
    },
    {
      "Category": "CITY PLANNING",
      "Definition": "References to changes to the built environment of cities and to urban policies shaping the city more broadly.",
      "Clarification": "Any sort of city planning measures are always coded as CITY PLANNING. This also includes measures taken to affect the choice of transport modes in cities, such as bike lanes, traffic calming measures, and restricting the access of cars to cities. Use CITY PLANNING for measures to make cities more resilient to changing weather.",
      "Negative Clarification": "Code wider traffic policy measures that happen to affect cities in a specific way as TRANSPORT. Code measures to make cities more resilient to specific extreme weather events as EXTREME WEATHER PROTECTION.",
      "Positive Example": "\"Experts point to Amsterdam's efforts to restrict cars to thoroughfares and prioritise bikes in the city as a model for German cities\". While the sentence focuses on transport, the measures suggested are urban planning policies.",
      "Negative Example": "\"The mayor's master plan to improve drainage capacities should help prevent flooding in the face of increasingly severe thunderstorms\". Severe thunderstorms are not a part of broad climate change, but specific and extreme events. Code as EXTREME WEATHER PROTECTION."
    },
    {
      "Category": "AGRICULTURE/FOOD",
      "Definition": "References to agricultural policy, the production, and the consumption of food.",
      "Clarification": "Includes any sort of government support or subsidies for industrial, traditional, or sustainable agriculture. Also includes regulations on farmers, changes in the goals of agricultural policies. Code political pushes to promote vegetarianism, veganism, or sustainable food practices, as well as opposition to such pushes.",
      "Negative Clarification": "Do not code purely personal appeals or statements (e.g. individuals committing to more sustainable food practices or reports of farmers acting more sustainably without government input).",
      "Positive Example": "\"Student activists protested against plans to abolish meat in all university cafeterias\". This is a political claim towards a push to more sustainable food choices.",
      "Negative Example": "\"The farmer, who also is an MP for the Green Party, switched to organic production using native strands of wheat\". While this is about sustainable agriculture, there is no political claim being made. Do not code at all."
    },
    {
      "Category": "ECONOMIC STRUCTURE",
      "Definition": "References to a general change in trading and economics in order to achieve a more sustainable economy.",
      "Clarification": "This category refers to the general direction of the economy and stances taken on the trade-off between economic growth and a greener economy. Also use for stances on the trade-off between existing industries and potential new industries in green technologies. Code references to alternative economic concepts (circular economy, donut economy, anti-capitalism etc.) in this category.",
      "Negative Clarification": "Do not include references to international vs. local trade under this category. Use INTERNATIONAL COOPERATION instead. Code references to research that are not directly linked to economic prospects under SCIENCE AND TECHNOLOGY.",
      "Positive Example": "\"Özdemir warns against a focus on the current car industry: 'Investing in green technologies now is the way to position Baden-Württemberg for the future'\". The speaker refers to the kind of future jobs he wants to create in Baden-Württemberg and takes a stance on the trade-off between existing industries and green jobs.",
      "Negative Example": "\"The Greens see investment in green innovation as a crucial step to be able to address the climate crisis in the future\". Here, green innovation is not linked to the economy, but to the climate itself. Code as SCIENCE AND TECHNOLOGY."
    },
    {
      "Category": "FINANCES",
      "Definition": "References to tax policy, government finances, and how to finance climate action.",
      "Clarification": "Applies to both general tax policy and specific taxes used to fight climate change. Use this category for stances on the trade-off between government investment in climate action and maintaining balanced budgets. Use this category for specific instruments to finance climate action (e.g. climate funds).",
      "Negative Clarification": "Code specific fees or levies for carbon emissions (e.g. carbon taxes, cap-and-trade schemes) as EMISSIONS. Code subsidies for specific sectors with the category relating to the relevant sector, unless the sentence is explicitly about how to pay for them.",
      "Positive Example": "\"The Greens demand an exemption from the debt brake for a 500 billion Euro investment in renewable energies\". The focus of the sentence is on whether to use public debt to finance renewables, rather than the financing itself.",
      "Negative Example": "\"A Le Pen government would lead to a shift in agricultural subsidies away from sustainable farming towards conventional farmers\". The sentence mentions government finances, but the focus is on the sector rather than the financing. Use AGRICULTURE/FOOD."
    },
    {
      "Category": "SCIENCE AND TECHNOLOGY",
      "Definition": "Claims that support or oppose climate research and innovation.",
      "Clarification": "Includes any sort of support or opposition to climate research, both material and verbal support. Also includes doubt on the credibility or motivations of climate scientists, and calls to \"follow the science\".",
      "Negative Clarification": "Does not include support for specific uses of existing technology. If the focus is on innovative industries (\"greentech\") as an economic opportunity, use ECONOMIC STRUCTURE. If the focus is on subsidising the implementation of new technologies, use the category for the appropriate sector.",
      "Positive Example": "\"The AfD dismissed the IPCC's latest report as 'alarmist nonsense to distract from the real problems'\". The sentence paints climate science as unreliable and is coded as opposition to SCIENCE AND TECHNOLOGY.",
      "Negative Example": "\"Green technologies are the future of our industry\", the PM said in his speech. There is no clear reference to research, and the focus is on industry implementations rather than new innovations. Code as ECONOMIC STRUCTURE."
    },
    {
      "Category": "EXTREME WEATHER PROTECTION",
      "Definition": "References to measures taken to protect people from extreme weather events.",
      "Clarification": "Extreme weather events are events that are unusual or severe, and pose a direct danger to people. Protection can mean the prevention of such events, measures taken to protect people from the harms of an extreme weather event, or relief efforts after an event has already happened.",
      "Negative Clarification": "Events that are part of everyday weather fluctuation are not part of extreme weather protection even if they have negative effects for people. Use a category related to the measure taken instead.",
      "Positive Example": "\"During his visit to the flood regions, Ciolacu promised a quick relief effort from the government\". Flooding is an unusual event with direct risks to people, and delivering relief is a part of protection.",
      "Negative Example": "\"Experts highlight the need for cities to create and maintain green spaces to prevent urban heat islands\". Urban heat islands are an effect of regular hot days, and not an extreme weather event. Use CITY PLANNING."
    },
    {
      "Category": "INTERNATIONAL COOPERATION",
      "Definition": "References to international agreements, trade, and the international aspects of climate and energy.",
      "Clarification": "Cooperation is understood broadly to include all international interactions, including international trade, migration, and references to international solidarity. Use INTERNATIONAL COOPERATION also for stances taken on more or less global interdependence in climate-related matters, such as energy imports.",
      "Negative Clarification": "Code only instances where the cooperation is relevant to national politics, e.g. by involving a national political actor. Do not code instances where the international cooperation is not related to climate or energy. Issues framed as national security issues are coded as SECURITY.",
      "Positive Example":  "\"The German Greens demand that the asylum laws are adapted to account for increased climate migration from dry regions in Africa\". Migration is an international issue, and it is related to climate in this case.",
      "Negative Example": "\"The President highlighted the role French nuclear power can play in making Europe less dependent on Russian gas and oil\". Dependency on Russian oil and gas is a question of national security. Use SECURITY."
    },
    {
      "Category": "CONSTRUCTION",
      "Definition": "References to the construction and renovation of buildings and infrastructure to make them more sustainable.",
      "Clarification": "This includes both legislative changes and demands to make new buildings more sustainable, and ones to improve the sustainability of existing buildings and infrastructure. Also use CONSTRUCTION for references to improving the sustainability of the construction process itself.",
      "Negative Clarification": "CONSTRUCTION refers to measures taken at the level of individual buildings. Use CITY PLANNING for references to measures that affect entire neighbourhoods or cities.",
      "Positive Example": "\"The proposed law would force new buildings to be fitted with heat pumps, and make homeowners replace existing oil heatings in the next ten years\". This is a change to the way individual buildings are constructed.",
      "Negative Example": "\"The plan for the neighbourhood involves water features to cool the area, and narrow streets that incentivise residents to walk and bike around the neighbourhood\". While these are infrastructure measures, they apply to the neighbourhood level rather than to each building. Use CITY PLANNING here."
    },
    {
      "Category": "GOVERNANCE",
      "Definition": "References to the way in which climate action should be enforced, and to the general way politics are conducted.",
      "Clarification": "Enforcement of climate action refers to claims about technocracy, personal freedom and responsibility. General political points refer to questions of power and democracy. Use GOVERNANCE for references to lobbying, populism, government overreach in the name of climate, etc.",
      "Negative Clarification": "References to technocracy should specifically be linked to the government. Use SCIENCE AND TECHNOLOGY for general references to experts. Code references to who should be held responsible as CLIMATE JUSTICE.",
      "Positive Example": "\"AfD MPs denounced the Heating Act as Green ideologues forcing their ideals onto the population\". The sentence takes a stance on personal choices, and claims that the government is overreaching.",
      "Negative Example": "\"The responsibility to address this crisis needs to fall onto big polluters, not the working man\", the union speaker said. Assignment of blame and burdens is coded as CLIMATE JUSTICE."
    },
    {
      "Category": "INDUSTRY",
      "Definition": "References to adjustments made in specific industries that do not have their own category, or across various sectors.",
      "Clarification": "Cross-sector measures can include references to manufacturing, industry, etc. Use INDUSTRY for measures that apply to specific sectors as well, such as the impact of tourism, greenifying supply specific supply chains, etc.",
      "Negative Clarification": "Where specific categories exist (AGRICULTURE/FOOD, TRANSPORT, CONSTRUCTION), use specific categories instead. If the emphasis is maintaining or creating jobs, use ECONOMIC STRUCTURE instead. If the emphasis is on compensation for expenses, use COMPENSATION.",
      "Positive Example": "\"The protestors called out the role of the steel sector as a major polluter and called for measures lowering its emissions\". The steel sector does not have its own specific category.",
      "Negative Example": "\"The government considers jobs in the car industry vital to Germany's economy\". The emphasis is on jobs and the role of cars for the German economy. Use ECONOMIC STRUCTURE."
    },
    {
      "Category": "PUBLIC HEALTH",
      "Definition": "References to the public health consequences of climate change.",
      "Clarification": "This includes both direct effects of pollution and emissions, and the more indirect health impacts of climate change. Also includes health-related issues such as pollution and sanitation.",
      "Negative Clarification": "Do not include references to pollution that are primarily about the effects on non-human environment. Use GENERAL ENVIRONMENT instead.",
      "Positive Example": "\"Experts urge government to address heat waves, or risk deaths among elderly\". Heat waves are a consequence of climate change, and public health is used as a reasoning for climate action.",
      "Negative Example": "\"The Greenpeace report points out further harms of coal plants, such as the major pollution of the air around the site\". While air pollution is a public health risk, it is not specifically linked to health effects. Use GENERAL ENVIRONMENT."
    },
    {
      "Category": "CLIMATE JUSTICE",
      "Definition": "References to the fair distribution of the costs of climate action and the consequences of climate change.",
      "Clarification": "The distribution of costs can refer both to the distribution of financial costs of climate action, and of negative effects such as having to change their behaviour. Also use CLIMATE JUSTICE for references to responsibility and assignments of blame.",
      "Negative Clarification": "Typically CLIMATE JUSTICE should be tied to some kind of group. References to personal freedom in general are coded as GOVERNANCE. Code references to financial compensation of victims of climate change as COMPENSATION.",
      "Positive Example": "\"Extinction Rebellion accuses billionaires of 'stealing the future of young people'\". There is an assignment of blame, a perpetrator, and a deserving group.",
      "Negative Example": "\"Protestors characterise the government's policies as oppressive and overreaching\". There is no reference to justice or fairness for a specific group. General discussions of government action are coded as GOVERNANCE."
    },
    {
      "Category": "COMPENSATION",
      "Definition": "References to or claims to compensate specific groups for climate or energy related costs.",
      "Clarification": "Compensation needs to be tied to directly to climate change or energy. The compensation can be broad, such as lowering energy costs for citizens, or specific to a group, such as compensating industries for necessary adaptations.",
      "Negative Clarification": "Compensation needs to happen after the fact, rather than be aimed at steering behaviour. Subsidies aimed at promoting specific industries are coded with the respective industry (FOSSIL FUELS, RENEWABLES, AGRICULTURE/FOOD, etc), or as INDUSTRY.",
      "Positive Example": "\"The Chancellor announced a credit of total 100 million Euro to help affected businesses adapt to new sustainability standards\". The compensation is applied after the fact to those affected by a policy, rather than being a financial incentive.",
      "Negative Example": "\"The government's plan promises up to 5000 Euro for home owners who want to install solar panels on their roof\". Here, money is used as an incentive for future actions (promoting solar energy). Use RENEWABLES instead."
    },
    {
      "Category": "GENERAL ENVIRONMENT",
      "Definition": "References to effects of climate change or climate action on non-human nature.",
      "Clarification": "The issue can also affect humans, but the reference should focus on non-human nature first. Effects on nature can be broad (e.g. melting glaciers) or specific to individual areas or species (e.g. a local loss of biodiversity).",
      "Negative Clarification": "Do not code harms to nature that are not related to climate change. If the focus of the sentence is on harms to humans, use other categories where possible.",
      "Positive Example": "\"The town's mayor opposed the wind farm, fearing it would disturb local rare bird populations\". The reference to bird populations is a non-climate claim that is related to climate action by the wind farm.",
      "Negative Example": "\"Experts urged for tighter water use rules during droughts to prevent the risk of forest fires threatening the community\". The focus is on forest fires as a threat to humans. Use EXTREME WEATHER PROTECTION."
    },
    {
      "Category": "SECURITY",
      "Definition": "References to protection against national and international security threats, either direct or indirect.",
      "Clarification": "Security refers to both direct responses to other countries' actions, as well as preventive action against potential threats to the population (e.g. energy diversification). Use SECURITY for general questions of energy security.",
      "Negative Clarification": "Security risks focus on man-made (political) and energy security threats, not threats by natural events. For protection against natural risks, use EXTREME WEATHER PROTECTION.",
      "Positive Example": "\"The government hopes that its renewable energy initiative will reduce dependency on Russian gas\". Reducing dependency on foreign energy sources is a preventative measure against a human security threat.",
      "Negative Example": "\"We are developing new early warning systems to protect the villages against the threat of flooding\". While the sentence speaks about protection from threats, the threat is a natural disaster risk. Use EXTREME WEATHER PROTECTION."
    }
  ]

sysprompt = f'''
You are an expert coder of political content in newspaper texts. You have a strong knowledge of British politics and know the relevant actors in the UK in 2025/26.
Your task is to read articles published in British newspapers and identify political claims made by nationally relevant actors, using the codebook below.

## Coding Instructions

A grammatical sentence can contain zero, one, or multiple relations that can be coded as core sentences. Core sentences express a relation between an subject (actor) and a target (either another actor or an issue). Subject and target are linked by a verb expressing some form of action. Code all core sentences you can find in a sentence, using the variables below. Core sentences can appear in multiple ways and patterns:

{cs_instructions}

The following variables are the basic building blocks of a core sentence. Every core sentence always needs a type, a subject, a direction, and at least one of object and/or issue.

The type of core sentence largely depends on the target of the claim being made. Use one of the following two categories:
{types}
{corevars}
issue_category:
Issue categories group issues further into one of 21 categories. These categories serve as a higher-level aggregation to analyze the climate-related agenda across countries. Use one of the following categories:

{issue_cats}

## Input Format

You will be given up to five sentences from a British newspaper article published between November 2025 and February 2026. One sentence is marked with > <. Code only the marked sentence, and use the rest of the text as context.

## Output Format

Return a JSON dictionary using the following structure:

{response_scheme}

Do not wrap in Markdown, and return nothing else.
'''

## Inference ##
GPTSMALL = 'gpt-oss:20b'
GPTLARGE = 'gpt-oss:120b'
GPTCLOUD = 'gpt-oss:120b-cloud'
GEMMA_CLOUD = 'gemma4:31b-cloud'
QWEN = 'qwen3.5:cloud'

modelname = QWEN

ollama.pull(modelname)

client = ollama.Client()
print("Ollama client loaded. Beginning inference now.")

# Full codebook prompt
out = []
ctr = 1

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
        format = response_scheme
    )
    
    ctr = ctr+1
    print(f"{ctr} texts analyzed.")

    out.append(response)

#Save output
transform_and_save(out, "output_cb_qwen.json")