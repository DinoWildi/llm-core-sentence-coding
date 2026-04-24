# !!!!! FIRST GRAB FUNCTIONS & PACKAGES FROM VALIDATION.R !!!!!

#### Reading in ####

fp <- path("C:/Users/dino1/Documents/GitHub/llm-core-sentence-coding/output")
res_base <- fromJSON(paste0(fp, "\\output_short_gpt-oss_120b.json")) %>% 
  json_transform() %>% 
  mutate(source = "llm")
res_codebook <- fromJSON(paste0(fp, "\\output_baseline_gpt-oss_120b.json")) %>% 
  json_transform() %>% 
  mutate(source = "llm")
res_fewshot <- fromJSON(paste0(fp, "\\output_fewshot_gptlarge_v1.json")) %>% 
  json_transform() %>% 
  mutate(source = "llm")

## GPT-OSS 120b: codebook

long_cb <- bind_rows(val, res_codebook) %>% 
  mutate(sentence = fuzzygroup(sentence))

long_cb <- group_by(long_cb, sentence) %>% 
  mutate(id = max(id, na.rm = T)) %>% 
  ungroup()

problems <- filter(long_cb, id == -Inf)
long_cb <- filter(long_cb, id != -Inf) %>% 
  arrange(id)

## GPT-OSS 120b: Shortened

long_sh <- bind_rows(val, res_base) %>% 
  mutate(sentence = fuzzygroup(sentence))

long_sh <- group_by(long_sh, sentence) %>% 
  mutate(id = max(id, na.rm = T)) %>% 
  ungroup()

problems <- filter(long_sh, id == -Inf)
long_sh <- filter(long_sh, id != -Inf) %>% 
  arrange(id)

## GPT-OSS 120b: Fewshot

long_fs <- bind_rows(val, res_fewshot) %>% 
  mutate(sentence = fuzzygroup(sentence))

long_fs <- group_by(long_fs, sentence) %>% 
  mutate(id = max(id, na.rm = T)) %>% 
  ungroup()

problems <- filter(long_fs, id == -Inf)
long_fs <- filter(long_fs, id != -Inf) %>% 
  arrange(id)
  
#### Core sentence detection (auto) ####

##Binary yes/no
cbmat <- binary_eval(long_cb, "matrix") %>% 
  mutate(prompt = "baseline")
shmat <- binary_eval(long_sh, "matrix") %>% 
  mutate(prompt = "shortened")
fsmat <- binary_eval(long_fs, "matrix") %>% 
  mutate(prompt = "fewshot")
binmat <- bind_rows(cbmat, shmat, fsmat)
binary_eval(long_cb, "f1")
binary_eval(long_sh, "f1")
binary_eval(long_fs, "f1")

##Number (plot)
val <- mutate(val, prompttype = "human")
long_cb <- mutate(long_cb, prompttype = "baseline")
long_sh <- mutate(long_sh, prompttype = "shortened")
long_fs <- mutate(long_fs, prompttype = "fewshot")

omnibus <- bind_rows(val, long_cb, long_sh, long_fs)

omniplot <- select(omnibus, id, sentence, prompttype, type) %>% 
  group_by(id, prompttype) %>%
  filter(!is.na(type)) %>% 
  summarise(number = n())

omniplot <- left_join(select(omnibus, id, prompttype), omniplot) %>% 
  mutate(number = if_else(is.na(number), 0, number)) %>% 
  unique() %>% 
  group_by(id) %>% 
  mutate(true = number[prompttype == "human"]) %>% 
  mutate(diff = number - true)

ggplot(filter(omniplot, prompttype != "human")) +
  geom_bar(aes(x = diff,
               fill = prompttype),
           position="dodge") +
  labs(x = "Difference between human coders and GPT-OSS 120b",
       y = "Number of grammatical sentences") +
  theme_minimal() +
  theme(legend.position = "bottom")

##Type
#Codebook
type_cb <- select(long_cb, id, sentence, type, source) %>% 
  mutate(type = str_replace_all(type, "_", "-")) %>% 
  reshaper("type") %>% 
  filter(!is.na(type)) %>% 
  mutate(across(c(true, llm),
                ~ if_else(is.na(.x), 0, .x)))
type_cb <- f1(type_cb, "type", output = "matrix") %>% 
  pivot_longer(cols = TP:FN,
               names_to = "result",
               values_to = "freq")
type_cb <- group_by(type_cb, type, result) %>% 
  summarise(n = sum(freq))

#Shortened
type_sh <- select(long_sh, id, sentence, type, source) %>% 
  mutate(type = str_replace_all(type, "_", "-")) %>% 
  reshaper("type") %>% 
  filter(!is.na(type)) %>% 
  mutate(across(c(true, llm),
                ~ if_else(is.na(.x), 0, .x)))
f1(type_sh, "type")

#Fewshot
type_fs <- select(long_fs, id, sentence, type, source) %>% 
  mutate(type = str_replace_all(type, "_", "-")) %>% 
  reshaper("type") %>% 
  filter(!is.na(type)) %>% 
  mutate(across(c(true, llm),
                ~ if_else(is.na(.x), 0, .x)))
f1(type_fs, "type")


#### Manual validation ####

manval <- read_csv2(path("../../data/manval_ct.csv")) %>% 
  filter(source != "true")

## Purely percentage-based among true CS

manval_true <- filter(manval, cs_id < 74)

manval_plot <- pivot_longer(manval_true, 
                            cols = contains("match"),
                            names_to = "measure") %>% 
  select(id, cs_id, sentence, prompttype, measure, value) %>% 
  mutate(value = if_else(value == 2, 1, value)) %>%
  mutate(value = if_else(value == 99, 0, value))

manval_plot <- group_by(manval_plot, prompttype, measure, value) %>% 
  summarise(matches = n()) %>%
  mutate(value = if_else(value == 0, "notmatched", "matched")) %>% 
  pivot_wider(names_from = "value",
              values_from = "matches")

manval_plot <- filter(manval_plot,
                      measure != "match_det" & measure != "match_whole") %>% 
  mutate(match_prop = round(100 * (matched/(notmatched + matched)), 3))

ggplot(manval_plot) +
  geom_bar(aes(x = measure,
               y = match_prop,
               fill = prompttype),
           stat = "identity",
           position = position_dodge2(padding = .15),
           width = .3) +
  labs(x = "Variable",
       y = "Proportion of correct codes",
       fill = "Type of prompt") +
  scale_x_discrete(limits = rev,
                   labels = c("Subject", "Object", "Issue", "Direction")) +
  theme_minimal() +
  theme(legend.position = "bottom")

#Presentation version

ggplot(manval_plot) +
  geom_bar(aes(x = match_prop,
               y = measure,
               fill = prompttype),
           stat = "identity",
           position = position_dodge2(padding = .15),
           width = .3) +
  labs(x = "% of correct codes",
       y = "Variable",
       fill = "Prompt") +
  scale_y_discrete(labels = c("Direction", "Issue", "Object", "Subject")) +
  scale_x_continuous(breaks = seq(0, 100, 10)) +
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
