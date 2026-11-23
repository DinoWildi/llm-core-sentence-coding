library(tidyverse)
library(semgram)
library(spacyr)

#Need to initialise Spacy once: Python process begins in background
spacy_initialize()

test <- "Friedrich Merz criticised the SPD leadership for its lacking commitment to necessary reforms. He said: 'Klingbeil's attempt to undermine agreements threatens the entire coalition'. His statements were rejected by the Vice Chancellor, who said he will not cease to defend the interests of his voters."

## Example workflow

#Parse sentence with Spacy:
test.dt <- spacy_parse(test, dependency = T)
test.dt

#Semantic parsing with semgram:
extract <- extract_motifs(test.dt,
                          pron_as_ap = F)
extract
