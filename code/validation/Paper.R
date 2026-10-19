# !!!!! FIRST GRAB FUNCTIONS & PACKAGES FROM VALIDATION.R !!!!!

#### Reading in ####

fp <- path("C:/Users/dino1/Documents/GitHub/llm-core-sentence-coding/output")
gemma <- fromJSON(paste0(fp, "\\output_cb_gemma.json")) %>% 
  json_transform() %>% 
  mutate(source = "llm")
gpt <- fromJSON(paste0(fp, "\\output_cb_gptlarge_2.json")) %>% 
  json_transform() %>% 
  mutate(source = "llm")


## GPT-OSS 120b: codebook

long_gemma <- bind_rows(val, gemma) %>% 
  mutate(sentence = fuzzygroup(sentence),
         model = "gemma")

long_gemma <- group_by(long_gemma, sentence) %>% 
  mutate(id = max(id, na.rm = T)) %>% 
  ungroup()

problems <- filter(long_gemma, id == -Inf)
long_gemma <- filter(long_gemma, id != -Inf) %>% 
  arrange(id)

## GPT-OSS 120b: CB

long_gpt <- bind_rows(val, gpt) %>% 
  mutate(sentence = fuzzygroup(sentence),
         model = "gpt")

long_gpt <- group_by(long_gpt, sentence) %>% 
  mutate(id = max(id, na.rm = T)) %>% 
  ungroup()

problems <- filter(long_gpt, id == -Inf)
long_gpt <- filter(long_gpt, id != -Inf) %>% 
  arrange(id)

  
#### Core sentence detection (auto) ####

##Binary yes/no
gemma_mat <- binary_eval(long_gemma, "matrix") %>% 
  mutate(model = "gemma")
gpt_mat <- binary_eval(long_gpt, "matrix") %>% 
  mutate(model = "gpt")
binmat <- bind_rows(gemma_mat, gpt_mat)

binmat <- mutate(binmat, type = case_when(
  y_true == 1 & y_pred == 1 ~ "tp",
  y_true == 0 & y_pred == 1 ~ "fp",
  y_true == 0 & y_pred == 0 ~ "tn",
  y_true == 1 & y_pred == 0 ~ "fn"
)) 

binmat <- select(binmat, model, type, Freq) %>% 
  pivot_wider(names_from = "type",
              values_from = "Freq")

binmat <- mutate(binmat,
                 precision = tp / (tp+fp),
                 recall = tp / (tp+fn)) %>% 
  mutate(f1 = 2*((precision*recall)/(precision+recall)))

pres_tab <- select(binmat, model, precision, recall, f1)
xtable::xtable(pres_tab)

binary_eval(long_gemma, "f1")
binary_eval(long_gpt, "f1")

##Number 

nr_gpt <- mutate(long_gpt, cs = if_else(is.na(type), 0, 1)) %>% 
  group_by(id, source, model) %>% 
  summarise(n_cs = sum(cs))


nr_gemma <- mutate(long_gemma, cs = if_else(is.na(type), 0, 1)) %>% 
  group_by(id, source, model) %>% 
  summarise(n_cs = sum(cs))
nr <- bind_rows(nr_gemma, nr_gpt) %>% 
  pivot_wider(names_from = source,
              values_from = n_cs) %>% 
  mutate(across(true:llm,
                ~ if_else(is.na(.x), 0, .x)))

f1(nr, "model")

##(plot)
val <- mutate(val, model = "human")
long_gpt <- mutate(long_gpt, model = "gpt")
long_gemma <- mutate(long_gemma, model = "gemma")

omnibus <- bind_rows(val, long_gemma, long_gpt)

omniplot <- select(omnibus, id, sentence, model, type) %>% 
  group_by(id, model) %>%
  filter(!is.na(type)) %>% 
  summarise(number = n())

omniplot <- left_join(select(omnibus, id, model), omniplot) %>% 
  mutate(number = if_else(is.na(number), 0, number)) %>% 
  unique() %>% 
  group_by(id) %>% 
  mutate(true = number[model == "human"]) %>% 
  mutate(diff = number - true)

ggplot(filter(omniplot, model != "human")) +
  geom_bar(aes(x = diff,
               fill = model),
           position="dodge") +
  labs(x = "Difference between human coders and GPT-OSS 120b",
       y = "Number of grammatical sentences") +
  theme_minimal() +
  theme(legend.position = "bottom")

##Type
type_gemma <- select(long_gemma, id, sentence, type, source) %>% 
  reshaper("type") %>% 
  filter(!is.na(type)) %>% 
  mutate(across(c(true, llm),
                ~ if_else(is.na(.x), 0, .x)))
type_gemma <- f1(type_gemma, "type", output = "matrix") %>% 
  pivot_longer(cols = TP:FN,
               names_to = "result",
               values_to = "freq")
type_gemma <- group_by(type_gemma, type, result) %>% 
  summarise(n = sum(freq))
f1(type_gemma, "type")

#GPT
type_gpt <- select(long_gpt, id, sentence, type, source) %>%
  reshaper("type") %>% 
  filter(!is.na(type)) %>% 
  mutate(across(c(true, llm),
                ~ if_else(is.na(.x), 0, .x)))
f1(type_gpt, "type")

#Fewshot
type_fs <- select(long_fs, id, sentence, type, source) %>% 
  mutate(type = str_replace_all(type, "_", "-")) %>% 
  reshaper("type") %>% 
  filter(!is.na(type)) %>% 
  mutate(across(c(true, llm),
                ~ if_else(is.na(.x), 0, .x)))
f1(type_fs, "type")


#### Manual validation ####

manval_gpt <- read_csv2(path("../../output/gpt_cb_v2_validated.csv")) %>% 
  mutate(cs_id = case_when(
    cs_id < 74 ~ cs_id,
    cs_id == 99 & is.na(type) ~ 0,
    cs_id == 99 & is.na(type) == F ~ 99)) %>% 
  mutate(across(.cols = contains("match"),
                ~ if_else(is.na(.x), 0, .x)))

manval_gemma <- read_csv2(path("../../output/gemma_cb_v2_validated.csv")) %>% 
  mutate(cs_id = case_when(
    cs_id < 74 ~ cs_id,
    cs_id == 99 & is.na(type) ~ 0,
    cs_id == 99 & is.na(type) == F ~ 99)) %>% 
  mutate(across(.cols = contains("match"),
                ~ if_else(is.na(.x), 0, .x)))

manval <- bind_rows(manval_gpt, manval_gemma)

## Purely percentage-based among true CS

manval_true <- filter(manval, cs_id < 74 & cs_id > 0)

manval_true <- mutate(manval_true, match_whole_auto = if_else(
  match_type == 1 & match_subject == 1 & match_object == 1 & match_direction == 1 & match_cat == 1,
  1, 0
))

manval_plot <- pivot_longer(manval_true, 
                            cols = contains("match"),
                            names_to = "measure") %>% 
  select(cs_id, sentence, model, measure, value)

#add missing true CS
all_measures <- unique(manval_plot$measure)
complete_df <- expand.grid(
  cs_id = 1:73,
  model = unique(manval_plot$model),
  measure = all_measures
)

manval_plot <- left_join(complete_df, manval_plot) %>% 
  arrange(model, cs_id) %>% 
  mutate(value = if_else(is.na(value), 0, value))


manval_plot <- group_by(manval_plot, model, measure, value) %>% 
  summarise(matches = n()) %>%
  mutate(value = if_else(value == 0, "notmatched", "matched")) %>% 
  pivot_wider(names_from = "value",
              values_from = "matches")

manval_plot <- filter(manval_plot,
                      measure %in% c("match_subject", 
                                     "match_object", 
                                     "match_direction", 
                                     "match_cat",
                                     "match_whole")) %>% 
  mutate(match_prop = round(100 * (matched/(notmatched + matched)), 3))

manval_plot <- mutate(manval_plot, measure = factor(measure,
                                                    levels = c("match_subject",
                                                               "match_object",
                                                               "match_direction",
                                                               "match_cat",
                                                               "match_whole"),
                                                    labels = c("Subject",
                                                               "Object",
                                                               "Direction",
                                                               "Issue category",
                                                               "Core sentence")))

ggplot(manval_plot) +
  geom_bar(aes(x = measure,
               y = match_prop,
               fill = model),
           stat = "identity",
           position = position_dodge2(padding = .15),
           width = .3) +
  labs(x = "Variable",
       y = "Proportion of correct codes",
       fill = "Model") +
  scale_fill_discrete(labels = c("Gemma 4 31B", "GPT-OSS 120B")) +
  scale_y_continuous(breaks = seq(0,80,10)) +
  theme_minimal() +
  theme(legend.position = "bottom")

#Presentation version

ggplot(manval_plot) +
  geom_bar(aes(x = match_prop,
               y = measure,
               fill = model),
           stat = "identity",
           position = position_dodge2(padding = .15),
           width = .3) +
  labs(x = "% of correct codes",
       y = "Variable",
       fill = "Model") +
  scale_fill_discrete(labels = c("Gemma 4", "GPT-OSS")) +
  scale_x_continuous(breaks = seq(0, 100, 10)) +
  scale_y_discrete(limits = rev) +
  theme_minimal() +
  theme(legend.position = "bottom",
        text = element_text(size = 15))

# Plotting binaries

binplot <- mutate(binmat, type = case_when(
  y_true == 1 & y_pred == 1 ~ "TP",
  y_true == 0 & y_pred == 1 ~ "FP",
  y_true == 1 & y_pred == 0 ~ "FN",
  TRUE ~ "TN")
) %>% 
  select(prompt, type, Freq) %>% 
  pivot_wider(names_from = "type",
              values_from = "Freq") %>% 
  mutate(precision = TP/(TP+FP),
         recall = TP/(TP+FN),
         f1 = (2*TP)/((2*TP)+FP+FN)) %>% 
  pivot_longer(cols = precision:f1,
               names_to = "metric",
               values_to = "value")

ggplot(binplot) +
  geom_bar(aes(x = value,
               y = metric,
               fill = prompt),
           stat = "identity",
           pos = position_dodge2(padding = .15),
           width = .4) +
  labs(x = NULL,
       y = NULL,
       fill = "Prompt type") +
  scale_x_continuous(breaks = seq(0, 1, .1)) +
  theme_minimal() +
  theme(legend.position = "bottom",
        text = element_text(size = 15))



#### Presentations Ireland ####

results <- tibble(
  model = c(rep("Gemma 4 31B", 6),
            rep("GPT-OSS 120B", 6)),
  measure = rep(c("Precision", "Recall", "F1"), 4),
  type = c(rep("Binary", 3), rep("Number", 3), rep("Binary", 3), rep("Number", 3)),
  value = c(.65, 1, .79,
            .53, .93, .68,
            .73, .96, .83,
            .57, .82, .67)
)

ggplot(results) +
  geom_bar(aes(x = value,
               y = measure,
               fill = model),
           stat = "identity",
           position = "dodge") +
  facet_wrap(~ type,
             ncol = 1) +
  labs(x = "",
       y = "",
       fill = "") +
  theme_minimal() +
  theme(text = element_text(size = 15),
        legend.position = "bottom")
