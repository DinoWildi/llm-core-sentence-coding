library(tidyverse)
library(quanteda)
library(fuzzyjoin)

#### Clean and prep UK Sentence frame ####

uk_articles <- read_csv(fs::path("../../data/UK training.csv"))

uk_articles <- mutate(uk_articles, content = str_replace_all(content, " \\| ", ". ")) %>% 
  mutate(content = str_replace_all(content, "  +|[^\\S ]", ". "))

uk_c <- corpus(uk_articles, docid_field = "ID", text_field = "content")

uk_sent <- corpus_reshape(uk_c, to = "sentences", docvars = T) %>% 
  convert(to = "data.frame")

uk_sent <- uk_sent %>% 
  mutate(text = str_remove_all(text, "^\\W+")) %>% 
  filter(str_detect(text, "\\d{1,2}.\\d{2}[a|p]m") == F) %>%  #removes timestamps in live articles
  mutate(text = str_remove_all(text, "\\.$"))
  
# compute context window: -3 / +1
uk_sent <- uk_sent %>%
  group_by(Headline) %>% 
  mutate(con_min1 = lag(text),
         con_min2 = lag(text, 2),
         con_min3 = lag(text, 3),
         con_plus1 = lead(text)) %>% 
  mutate(across(con_min1:con_plus1,
                ~ if_else(is.na(.x), "", .x)))

uk_sent <- mutate(uk_sent, contexted = paste(con_min3, 
                                             con_min2, 
                                             con_min1, 
                                             ">", text, "<",
                                             con_plus1,
                                             sep = " ")) %>% 
  mutate(sep_context = paste(paste("Text:", text, sep = " "),
                             paste("Context:", 
                                   con_min3, con_min2, con_min1, text, con_plus1, sep = " "),
                             sep = "\n")) %>% 
  mutate(across(contexted:sep_context,
                ~ str_replace_all(.x, " {2,}", " "))) %>% 
  mutate(contexted = str_remove_all(contexted, "^ "))

write_csv(uk_sent, "uk_sentences.csv")


#### Bringing in validation set ####

valset <- read_csv2("C:\\Users\\dino1\\Documents\\Academia\\D - Data\\L - LLM paper test\\Validation set LLM_new.csv")

# Fuzzy joining
sentences <- select(valset, sentence) %>% 
  unique()

positives <- stringdist_left_join(sentences, uk_sent,
                                  by = c(sentence = "text"),
                                  method = "osa",
                                  max_dist = 5)
is.na(positives$text) %>% table()

positives <- filter(positives, is.na(text) == F)

#### Adding in negatives ####

#safe negatives

uk_codes <- read_csv("C:\\Users\\dino1\\Documents\\Academia\\D - Data\\B - PCA handcoding\\Data\\uk_validation.csv")

uk_code_sents <- filter(uk_codes, user_name != "Marcial Marin" & 
                          user_name != "Luca Li Calzi") %>% 
  select(sentence_string) %>% 
  unique()

trueneg <- stringdist_anti_join(uk_sent, uk_code_sents,
                                by = c(text = "sentence_string"),
                                method = "osa",
                                max_dist = 7) %>% 
  filter(Headline != "Lammy says he was not ‘equipped with the details’ when facing questions on mistaken prisoner release at PMQs – UK politics live") %>% 
  filter(str_count(text, " ") > 7)

trueneg <- mutate(trueneg, text = str_replace_all(text, "www\\..*", "\\."))

negatives <- filter(trueneg, row_number() %in% sample(1:nrow(trueneg), 46))

cuts <- c(3, 21, 23, 31, 43, 46, 47, 49)

negatives <- filter(negatives, !(row_number() %in% cuts))

testset <- bind_rows(positives, negatives)
testset <- testset[sample(1:nrow(testset)), ]
testset <- mutate(testset, id = row_number()) %>% 
  select(id, text, contexted) %>% 
  mutate(contexted = str_remove_all(contexted, "(NA )+"))
  

validation <- stringdist_left_join(testset, valset,
                                   by = c(text = "sentence")) %>% 
  select(-sent_id)

save(testset, validation, file = "LLM validation.Rdata")
write_csv(testset, "UK_texts.csv")
