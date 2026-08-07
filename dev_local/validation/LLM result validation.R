library(tidyverse)
library(fs)
library(jsonlite)
library(stringdist)

#### Functions ####

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
id_match <- function(output, val = validation) {
  
  val_id <- validation %>% select(id, text) %>% 
    unique()
  out_id <- output %>% select(sentence) %>% 
    unique()
  
  ids <- fuzzyjoin::stringdist_right_join(val_id, out_id,
                                          by = c("text" = "sentence"), 
                                          method = "osa",
                                          max_dist = 5)
  ids <- select(ids, id, sentence)
  out <- left_join(ids, output) %>% 
    filter(!is.na(id))
  
  return(out)
}
nr_eval <- function(data_true, data_pred, truecol = type_cs, predcol = type) {
  true_nrs <- mutate(data_true, nr = if_else(is.na({{truecol}}), 0, 1)) %>% 
    group_by(id) %>% 
    summarise(nr_true = sum(nr))
  
  pred_nrs <- mutate(data_pred, nr = if_else(is.na({{predcol}}), 0, 1)) %>% 
    group_by(id) %>% 
    summarise(nr_llm = sum(nr))
  
  numbers <- inner_join(true_nrs, pred_nrs)
  
  numbers <- mutate(numbers, 
                    tp = if_else(nr_true <= nr_llm, nr_true, nr_llm),
                    fp = if_else(nr_llm > nr_true, nr_llm - nr_true, 0),
                    fn = if_else(nr_true > nr_llm, nr_true - nr_llm, 0))
  
  f1_nrs <- summarise(numbers, across(tp:fn, ~ sum(.x))) %>% 
    mutate(precision = tp/(tp+fp),
           recall = tp/(tp+fn),
           f1 = (2*precision*recall)/(precision+recall))
  
  return(f1_nrs)
}

var_eval_dep <- function(data_true, data_pred, true_col, pred_col, id_col_true = id, id_col_pred = id){
  
  lookup <- data_true %>% 
    select({{id_col_true}}, {{true_col}}) %>% 
    group_by({{id_col_true}}) %>% 
    summarise(labs_true = list({{true_col}}), .groups = "drop")
  
  tmp <- left_join(data_pred, lookup) %>% 
    mutate(match = ({{pred_col}} %in% unlist(labs_true)))
  
  out <- summarise(tmp,
                   precision = mean(match, na.rm = T))
  
  return(out)
}

var_eval <- function(data_true, data_pred, true_col, pred_col, id_col = id, f1_type = "micro") {
  
  true <- select(data_true, {{id_col}}, {{true_col}}) %>% 
    rename(label = {{true_col}}) %>% 
    mutate(type = "true")
  pred <- select(data_pred, {{id_col}}, {{pred_col}}) %>% 
    rename(label = {{pred_col}}) %>% 
    mutate(type = "predicted")
  
  data = bind_rows(true, pred)
  
  counts <- group_by(data, {{id_col}}, type, label) %>% 
    summarise(count = n(),
              .groups = "drop_last") %>% 
    pivot_wider(names_from = type,
                values_from = count) %>% 
    mutate(across(c(true, predicted),
                  ~ if_else(is.na(.x), 0, .x)))
  
  counts <- mutate(counts,
                   tp = min(true, predicted),
                   fp = predicted - tp,
                   fn = true - tp)
  
  if(f1_type == "micro"){
    f1 <- counts %>% 
      ungroup() %>%  
      summarise(across(tp:fn, ~ sum(.x))) %>% 
      mutate(precision = tp/(tp+fp),
             recall = tp/(tp+fn),
             f1 = (2*precision*recall)/(precision+recall))
  }
  
  if(f1_type == "perclass"){
    f1 <- counts %>% 
      group_by(label) %>%  
      summarise(across(tp:fn, ~ sum(.x))) %>% 
      mutate(precision = tp/(tp+fp),
             recall = tp/(tp+fn),
             f1 = (2*precision*recall)/(precision+recall)) %>% 
      ungroup() %>% 
      mutate(macro_f1 = mean(f1, na.rm = T))
    
    f1 <- mutate(f1, n_pred = tp+fp) %>% 
      arrange(desc(n_pred))
  }
  
  if(f1_type != "perclass" & f1_type != "micro"){
    stop("Invalid f1_type option. Use \"micro\" or \"perclass\"")
  }
  
  return(f1)
    
}
  
#### Load + clean validation set ####

#Gold standard dataset
load(path("../../data/LLM Validation.RData"))

#Minor cleaning

validation <- mutate(validation, sub_org = case_when(
  sub_org %in% c("Labour Party (UK)", "Labour") ~ "Labour",
  sub_org %in% c("E3G Thinktank", "E3G Thinkthank") ~ "E3G",
  sub_org == "Experts or scientists (unaffiliated" ~ "Experts or scientists (unaffiliated)",
  sub_org == "BP Plc / British Petrolium" ~ "BP",
  TRUE ~ sub_org
))

validation <- mutate(validation, obj_org = case_when(
  obj_org %in% c("Conservative party (Tories)", "Tories") ~ "Conservative Party",
  obj_org == "Labour Party (UK)" ~ "Labour",
  TRUE ~ obj_org
))

validation <- mutate(validation, issue_lv1 = if_else(
  issue_lv2 == "lowering energy costs", "COMPENSATION", issue_lv1
))

validation <- mutate(validation, issue_lv1 = if_else(
  issue_lv1 == "climate change (general)",
  "CLIMATE CHANGE (general)",
  str_to_upper(issue_lv1)
))
  
#### Load LLM output ####
initial <- json_transform(path(
  "C:/Users/dino1/Documents/GitHub/llm-core-sentence-coding/output/reviewer_test_initial_gemma.json")) %>% 
  id_match() %>% 
  arrange(id)

initial <- mutate(initial, subject_organisation = case_when(
  subject_organisation == "Labour Party" ~ "Labour",
  subject_organisation == "Independent/Expert" ~ "Experts or scientists (unaffiliated)",
  subject_organisation == "Energy Industry" ~ "Energy sector",
  TRUE ~ subject_organisation
))
initial <- mutate(initial, object_organisation = case_when(
  object_organisation == "Labour Party" ~ "Labour",
  object_organisation == "Independent/Expert" ~ "Experts or scientists (unaffiliated)",
  object_organisation == "Energy Industry" ~ "Energy sector",
  TRUE ~ object_organisation
))

reviewed <- json_transform(path(
  "C:/Users/dino1/Documents/GitHub/llm-core-sentence-coding/output/reviewer_test_gemma_gpt.json")) %>% 
  id_match() %>% 
  arrange(id)

reviewed <- mutate(reviewed, subject_organisation = case_when(
  subject_organisation == "Labour Party" ~ "Labour",
  subject_organisation == "Independent/Expert" ~ "Experts or scientists (unaffiliated)",
  subject_organisation == "Energy Industry" ~ "Energy sector",
  TRUE ~ subject_organisation
))
reviewed <- mutate(reviewed, object_organisation = case_when(
  object_organisation == "Labour Party" ~ "Labour",
  object_organisation == "Independent/Expert" ~ "Experts or scientists (unaffiliated)",
  object_organisation == "Energy Industry" ~ "Energy sector",
  TRUE ~ object_organisation
))

#### F1 for numbers ####

nr_eval(validation, initial)
nr_eval(validation, reviewed)

#### CS Variables ####

var_eval(validation, initial, sub_org, subject_organisation)
var_eval(validation, initial, obj_org, object_organisation)
var_eval(validation, initial, issue_lv1, issue_cat)
var_eval(validation, initial, dir, direction)

var_eval(validation, reviewed, sub_org, subject_organisation)
var_eval(validation, reviewed, obj_org, object_organisation)
var_eval(validation, reviewed, issue_lv1, issue_cat)
var_eval(validation, reviewed, dir, direction)

#### Reviewer architecture ####
#### 
raw_gemma <- json_transform(path(
  "C:/Users/dino1/Documents/GitHub/llm-core-sentence-coding/output/reviewer_test_initial.json")) %>% 
  id_match() %>% 
  arrange(id)

raw_gemma <- mutate(raw_gemma, subject_organisation = case_when(
    subject_organisation == "Labour Party" ~ "Labour",
    subject_organisation == "Independent/Expert" ~ "Experts or scientists (unaffiliated)",
    subject_organisation == "Energy Industry" ~ "Energy sector",
    TRUE ~ subject_organisation),
  object_organisation = case_when(
    object_organisation == "Labour Party" ~ "Labour",
    object_organisation == "Independent/Expert" ~ "Experts or scientists (unaffiliated)",
    object_organisation == "Energy Industry" ~ "Energy sector",
    TRUE ~ object_organisation
  )
)

gemma_reviewed <- mutate(gemma_reviewed, subject_organisation = case_when(
  subject_organisation == "Labour Party" ~ "Labour",
  subject_organisation == "Independent/Expert" ~ "Experts or scientists (unaffiliated)",
  subject_organisation == "Energy Industry" ~ "Energy sector",
  TRUE ~ subject_organisation),
  object_organisation = case_when(
    object_organisation == "Labour Party" ~ "Labour",
    object_organisation == "Independent/Expert" ~ "Experts or scientists (unaffiliated)",
    object_organisation == "Energy Industry" ~ "Energy sector",
    TRUE ~ object_organisation
  )
)

nr_eval(validation, raw_gemma)
nr_eval(validation, gemma_reviewed)

var_eval(validation, raw_gemma, sub_org, subject_organisation)
var_eval(validation, raw_gemma, obj_org, object_organisation)
var_eval(validation, raw_gemma, issue_lv1, issue_cat)
var_eval(validation, raw_gemma, dir, direction)

var_eval(validation, gemma_reviewed, sub_org, subject_organisation)
var_eval(validation, gemma_reviewed, obj_org, object_organisation)
var_eval(validation, gemma_reviewed, issue_lv1, issue_cat)
var_eval(validation, gemma_reviewed, dir, direction)
