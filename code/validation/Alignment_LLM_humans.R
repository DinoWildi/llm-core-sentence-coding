library(tidyverse)
library(krippendorffsalpha)
library(fs)
library(jsonlite)
library(fuzzyjoin)
library(stringdist)

#### Loading LLM results #### 
## requires function json_transform from LLM result validation.R
cb_gemma <- json_transform(path("../../output/UK Full Set/codebook_gemma4_31b.json")) |> 
  mutate(user_name = "gemma_cb")
cb_gpt <- json_transform(path("../../output/UK Full Set/codebook_gpt-oss_120b_fulluk.json")) |> 
  mutate(user_name = "gpt_cb")
rev_gemma <- json_transform(path("../../output/UK Full Set/reviewer_gemma_cloud.json")) |> 
  mutate(user_name = "reviewer_gemma")
entity_gemma <- json_transform(path("../../output/UK Full Set/entity_gemma4_ukfull.json")) |> 
  mutate(user_name = "entity_gemma")

llm_results <- bind_rows(cb_gemma, cb_gpt, rev_gemma, entity_gemma)
sentences <- select(llm_results, sentence) |> mutate(sentence = fuzzygroup(sentence)) |> unique()

#### Loading + cleaning manual coder data ####
coders_raw <- read_csv(path("../../data/uk_validation.csv")) 

coders <- coders_raw |> 
  select(user_name, sentence_string, type_cs_string, SUBJECT_ORGANIZATION_string, DIRECTION_string, object_organisation_string, issue_category_string) |> 
  rename_with(~ str_remove_all(.x, "_string.*")) |> 
  rename_with(~ str_to_lower(.x)) |> 
  rename(type = type_cs,
         subject_organisation = subject_organization,
         issue_cat = issue_category)

# Binding to LLM data

full <- bind_rows(coders, llm_results) |> 
  filter(user_name != "Marcial Marin") |> 
  unique()

full <- stringdist_inner_join(sentences, full)

full <- select(full, -sentence.y) |> 
  rename(sentence = sentence.x)



#### Reshaping function ####

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


#### Function for pairwise comparison ####

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

#### Running LLM models against coders #### 

### All models, pairwise, without Marcial

## Number of CS
nr_cs <- select(full, user_name, subject_organisation, sentence) %>% 
  mutate(nr = if_else(is.na(subject_organisation), 0, 1)) |> 
  group_by(sentence, user_name) |> 
  summarise(n_cs = sum(nr)) |>  
  ungroup()

matrix.full <- ka_matrix(nr_cs, "n_cs")
matrix.full[is.na(matrix.full)] <- 0
kanr <- krippendorffs.alpha(matrix.full, level = "ratio", control = list(parallel = F), confint = F, verbose = T)
full.pair <- ka.pairwise(matrix.full, "ratio")

# visual

coderorder1 <- select(full.pair, coder1) |> 
  summarise(n = n(),
            .by = coder1) |> 
  arrange(desc(n))
  
coderorder1 <- coderorder1$coder1

coderorder2 <- select(full.pair, coder2) |> 
  summarise(n = n(),
            .by = coder2) |> 
  arrange(desc(n))

coderorder2 <- coderorder2$coder2

ggplot(full.pair, aes(
  x = factor(coder1, levels = coderorder1),
  y = factor(coder2, levels = coderorder2))) +
  geom_tile(aes(fill = alpha)) +
  geom_text(aes(label = round(alpha, digits = 3))) +
  scale_fill_continuous(limits = c(1, -1),
    palette = c("#d00","#dd0","#0a0"))


## Presence of CS

presence <- mutate(nr_cs, presence = if_else(n_cs > 0, 1, 0))
presmat <- ka_matrix(presence, "presence")
presmat[is.na(presmat)] <- 0

krippendorffs.alpha(presmat, level = "nominal", control = list(parallel=F)) |> summary()
pres.pair <- ka.pairwise(presmat, "nominal")

# visual

coderorder1 <- select(pres.pair, coder1) |> 
  summarise(n = n(),
            .by = coder1) |> 
  arrange(desc(n))

coderorder1 <- coderorder1$coder1

coderorder2 <- select(pres.pair, coder2) |> 
  summarise(n = n(),
            .by = coder2) |> 
  arrange(desc(n))

coderorder2 <- coderorder2$coder2

ggplot(pres.pair, aes(
  x = factor(coder1, levels = coderorder1),
  y = factor(coder2, levels = coderorder2))) +
  geom_tile(aes(fill = alpha)) +
  geom_text(aes(label = round(alpha, digits = 3))) +
  scale_fill_continuous(limits = c(1, -1),
                        palette = c("#d00","#dd0","#0a0"))
