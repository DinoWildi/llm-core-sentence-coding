library(tidyverse)
library(fs)
library(jsonlite)
library(stringdist)
library(fuzzyjoin)
library(krippendorffsalpha)

coder_raw <- read_csv2(path("../../data/UK_handcoded_edited.csv"),
                       locale=locale(encoding="latin1"))

#### Functions ####

ka_matrix <- function(data, 
                      varcol, 
                      codercol="user_name", 
                      sentences="sentence"){
  
  coders <- unique(data[[codercol]])
  sents <- unique(data[[sentences]])
  
  out <- matrix(NA,
                nrow = length(sents),
                ncol = length(coders),
                dimnames = list(sents, coders))
  
  for(i in 1:nrow(data)){
    out[data[[sentences]][i], data[[codercol]][i]] <- data[[varcol]][i]
  }
  
  return(out)
}

# JSON function

json_transform <- function(path) {
  
  data <- fromJSON(path)
  
  data_unnested <- data %>% 
    unnest_wider(core_sents) %>% 
    unnest_longer(!sentence)
  
  sents <- select(data, sentence)
  
  out <- left_join(sents, data_unnested,
                   relationship = "many-to-many") %>% 
    mutate(type = str_replace_all(type, "_", "-"))
  
  return(out)
  
}

getmode <- function(data){
  clean <- na.omit(data)
  
  freq <- table(clean)
  
  modes <- names(freq[freq == max(freq)])
  
  if(length(modes) > 1){
    modes <- sample(modes, 1)
  }
  
  return(modes)
}



#### HUMAN CODERS ####
coders <- coder_raw |>
  group_by(sentence_id) |> 
  mutate(sentence = getmode(sentence_string)) |> 
  ungroup() |> 
  select(user_name, sentence, type_cs_string, SUBJECT_ORGANIZATION_string, DIRECTION_string, object_organisation_string, issue_category_group) |> 
  rename_with(~ str_remove_all(.x, "_string|_group")) |> 
  rename_with(tolower)

## Human alignment
## 

#Load all sentences for matching
fulltxt <- read_csv(path("../../data/uk_sentences.csv"))

sents <- select(fulltxt, text) |> 
  mutate(text = str_replace_all(text, " \\.$|\\.\\.$", "\\.")) |> 
  mutate(text = str_replace_all(text, ":\\.", ":")) |> 
  filter(str_count(text, " ") > 5)

# Link to human DS

merged <- stringdist_left_join(sents, coders, method = "osa", max_dist = 7, by = c(text = "sentence"))
merged <- group_by(merged, text) |> 
  mutate(sentence_id = cur_group_id()) |> 
  ungroup()

# Full dataset
fullmap <- merged |> 
  select(sentence_id, user_name) |> 
  expand(sentence_id, user_name) |> 
  filter(is.na(user_name) == F)

ds <- left_join(fullmap, merged)

# Calculate alignment

human_nr <- ds |>  
  mutate(cs_yn = if_else(is.na(type_cs), 0, 1)) |> 
  group_by(user_name, sentence_id) |> 
  summarise(n_cs = sum(cs_yn)) |> 
  mutate(presence = if_else(n_cs > 0, 1, 0))

ka_mat_nr <- select(human_nr, -presence) |> 
  pivot_wider(names_from = "user_name",
              values_from = "n_cs") |> 
  column_to_rownames(var = "sentence_id") |> 
  as.matrix()

ka_nr_human <- krippendorffs.alpha(ka_mat_nr,
                                   level = "ratio",
                                   method = "customary",
                                   verbose = T,
                                   control = list(
                                     parallel = F
                                   ))

summary(ka_nr_human)

ka_mat_presence <- select(human_nr, -n_cs) |> 
  pivot_wider(names_from = "user_name",
              values_from = "presence") |> 
  column_to_rownames(var = "sentence_id") |> 
  as.matrix()

ka_cs_human <- krippendorffs.alpha(ka_mat_presence,
                                   level = "nominal",
                                   method = "customary",
                                   verbose = T,
                                   control = list(
                                     parallel = F
                                   ))

summary(ka_nr_human)

# Pairwise comparison

ka.pairwise <- function(matrix, level){
  
  coders <- colnames(matrix)
  coderpairs <- combn(coders, 2, simplify = F) |> 
    map(~ list(coder1 = .x[1], coder2 = .x[2]))
  
  results <- map_dfr(coderpairs, ~ {
    tmp.mat <- matrix[, c(.x$coder1, .x$coder2)]
    tmp_ka <- krippendorffs.alpha(tmp.mat, level = level, confint = F)
    
    tibble(
      coder1 = as.character(.x[1]),
      coder2 = as.character(.x[2]),
      alpha = tmp_ka$alpha.hat)
  }
  )
  
  return(results)
}

presence_pairwise <- ka.pairwise(ka_mat_presence, "nominal")
nr_pairwise <- ka.pairwise(ka_mat_nr, "ratio")


#### Load LLMs ####


cb_gemma <- json_transform(path("../../output/UK Full Set/codebook_gemma4_31b.json"))
