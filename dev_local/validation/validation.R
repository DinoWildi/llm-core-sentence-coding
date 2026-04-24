library(tidyverse)
library(fs)
library(jsonlite)
library(fuzzyjoin)
library(stringdist)
library(MLmetrics)

#### Data loading and merging ####
# Load validation set 

load(path("../../data/LLM Validation.RData"))

# Load relevant JSON

json_transform <- function(data) {
  
  data_unnested <- data %>% 
    unnest_wider(core_sents) %>% 
    unnest_longer(!sentence)
  
  sents <- select(data, sentence)
  
  out <- left_join(sents, data_unnested) %>% 
    mutate(type = str_replace_all(type, "_", "-"))
  
  return(out)
  
}

results <- fromJSON(path("../../output/output_baseline_gpt-oss_120b.json")) 
results <- json_transform(results)

# Harmonizing colnames and merge to long frame

val <- select(validation,
              id,
              text,
              type_cs,
              sub_org,
              dir,
              obj_org,
              issue_lv2,
              alt_issue) %>% 
  mutate(source = "true") %>% 
  rename(sentence = text,
         type = type_cs,
         subject = sub_org,
         direction = dir,
         object = obj_org,
         issue = issue_lv2)

results <- mutate(results, source = "llm") %>% 
  mutate(type = str_replace_all(type, "_", "-"))
long <- bind_rows(val, results)

#### Edit and clean long dataset ####

## Defuzzy texts

fuzzygroup <- function(texts, threshold = 0.95, method = "osa"){
  
  validity = if_else(method %in% 
                       c("osa", "lv", "dl", "hamming", "lcs", "qgram", "cosine", "jaccard", "jw",
                         "soundex"),
                     1, 0)
  
  if(validity == 0){
    print("No valid method of computing string distance given. See ?stringdist for valid methods")
    return()
  }
  
  distmat <- stringdistmatrix(texts, texts, method = method)
  sim_matrix <- 1 - distmat / max(distmat)
  
  # Cluster similar texts
  clusters <- list()
  used <- rep(FALSE, length(texts))
  for (i in 1:length(texts)) {
    if (!used[i]) {
      cluster <- which(sim_matrix[i, ] >= threshold)
      clusters[[length(clusters) + 1]] <- cluster
      used[cluster] <- TRUE
    }
  }
  
  # Replace each text with the most common version in its cluster
  normalized_texts <- texts
  for (cluster in clusters) {
    # Get the frequency table for texts in the cluster
    freq_table <- table(texts[cluster])
    # Find the maximum frequency
    max_freq <- max(freq_table)
    # Get all texts with the maximum frequency
    candidates <- names(freq_table[freq_table == max_freq])
    
    # If there's only one candidate, use it
    if (length(candidates) == 1) {
      most_common <- candidates[1]
    } else {
      # If multiple candidates, choose the longest one
      most_common <- candidates[which.max(nchar(candidates))]
    }
    
    normalized_texts[cluster] <- most_common
  }
  
  return(normalized_texts)
}

long <- mutate(long, sentence = fuzzygroup(sentence, threshold = .8))

# harmonize IDs
long <- group_by(long, sentence) %>% 
 mutate(id = max(id, na.rm = T)) %>% 
 ungroup()
 
problems <- filter(long, id == -Inf)
long <- filter(long, id != -Inf) %>% 
  arrange(id)

#### Metrics functions ####

reshaper <- function(data, varcol){
  
  out <- select(data, id, source, .data[[varcol]]) %>% 
    summarise(count = n(),
              .by = c(id, source, .data[[varcol]])) %>% 
    pivot_wider(names_from = source,
                values_from = count)
  return(out)
}

f1 <- function(data, varcol, truecol = "true", evalcol = "llm", na.rm = T, output = "f1"){
  out <- data %>% 
    mutate(across(c(.data[[truecol]], .data[[evalcol]]),
                  ~ if_else(is.na(.x), 0, .x))) %>% 
    group_by(id, .data[[varcol]]) %>% 
    mutate(TP = if_else(.data[[evalcol]] <= .data[[truecol]], .data[[evalcol]], .data[[truecol]]),
           FP = max(.data[[evalcol]]-.data[[truecol]], 0),
           FN = max(.data[[truecol]]-.data[[evalcol]], 0)) %>%
    ungroup()
  
  if(na.rm == T) out <- filter(out, !is.na(.data[[varcol]]))
  
   
   out_f1 <- out %>% 
    group_by(.data[[varcol]]) %>% 
    summarise(across(TP:FN,
                     ~ sum(.x))) %>% 
    mutate(f1 = (2*TP)/(2*TP + FP + FN)) %>% 
    mutate(macro = mean(f1))
   
   if(output == "f1") return(out_f1)
   if(output == "matrix") return(out)
}

#### Metrics functions ####

# Binary yes/no

binary_eval <- function(long, target){
  bin <- mutate(long, binary = if_else(is.na(type), 0, 1))
  
  bin <- select(bin, id, source, binary) %>% 
    pivot_wider(names_from = source,
                values_from = binary,
                values_fn = ~ max(.x))
  
  if (target == "matrix") {
    out <- ConfusionDF(bin$llm, bin$true)
  } else if (target == "f1") {
    out <- F1_Score(bin$llm, bin$true, positive = 1)
  } else {stop("Unknown target, use matrix or f1")}
  
  return(out)
}


# Type among detected CS
type_cs <- reshaper(long, "type")
type_f1 <- f1(type_cs, "type")

# Direction among detected CS

dir <- filter(long, !is.na(type)) %>% 
  reshaper("direction") %>% 
  mutate(across(c(true, llm),
                ~ if_else(is.na(.x), 0, .x)))
dir <- filter(dir, direction != "ambiguous" & direction != "ambivalent")
dir_f1 <- f1(dir, "direction")

# Subjects

subj <- filter(long, !is.na(subject)) %>% 
  reshaper("subject")



#### Merging for paper ####

