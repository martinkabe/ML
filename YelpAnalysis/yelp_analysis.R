

# Loading Packages --------------------------------------------------------
library(jsonlite) # install.packages("jsonlite")
library(tidyverse) # install.packages("tidyverse")
library(dplyr) # install.packages("dplyr")
library(stringi) # install.packages("stringi")
library(maps) # install.packages("maps")
library(ggmap) # install.packages("ggmap")
library(ggplot2) # install.packages("ggplot2")
library(igraph) #  install.packages("igraph")
library(wordcloud) # install.packages("wordcloud")
library(tidytext) # install.packages("tidytext")
library(data.table) # install.packages("data.table")
library(DT) # install.packages('DT')

rm(list=ls())

fillColor = "#4c4cff"
fillColor2 = "#F1C40F"

# Business DataSet --------------------------------------------------------
biz <- stream_in(file("~/Projects/RScripts/Data/business.json"))

# top 10 Categories of Business
table(trimws(strsplit(unique(biz$categories), ",") %>% 
                                unlist())) %>% as.data.frame() %>% arrange(desc(Freq)) %>%
  rename(Category=Var1) %>% top_n(n = 10, wt = Freq) %>% 
ggplot(aes(x=Category, y=Freq)) +
  geom_bar(stat="identity", fill="steelblue") +
  geom_text(aes(label=Freq), vjust=1.6, color="white", size=2.6) +
  theme_minimal() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Top 10 Categories of Business") + xlab("Category") + ylab("Count")

# stars distribution for Restaurants
biz %>% filter(grepl("restaurants", tolower(categories))) %>%
  group_by(stars) %>% summarise(Count=n()) %>% ungroup() %>%
ggplot(aes(x=stars, y=Count)) +
  geom_bar(stat='identity', colour="white", fill = fillColor) +
  geom_text(aes(x = stars, y = 1, label = paste0("(",round(Count/1e3)," K )",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'white',
            fontface = 'bold') +
  labs(x = 'City', y = 'Count of stars Category', 
       title = 'Distribution of Stars Rating for Restaurants only') +
  coord_flip() + 
  theme_bw()

# stars less than 3 group by city
avg_stars <- biz %>% filter(grepl("restaurants", tolower(categories))) %>% 
  group_by(city) %>% 
  summarise(stars_avg=mean(stars, na.rm=T)) %>% ungroup() %>%
  filter(stars_avg < 3) %>% arrange(desc(stars_avg)) %>% top_n(10, stars_avg)

# Top Ten Cities with the most Business parties mentioned in Yelp
biz %>%
  group_by(city) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(City = reorder(city,Count)) %>%
  head(10) %>%
  ggplot(aes(x = City,y = Count)) +
  geom_bar(stat='identity',colour="white", fill = fillColor) +
  geom_text(aes(x = City, y = 1, label = paste0("(",round(Count/1e3)," K )",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'white',
            fontface = 'bold') +
  labs(x = 'City', y = 'Count of Reviews', 
       title = 'Top Ten Cities with the most Business parties in Yelp') +
  coord_flip() + 
  theme_bw()

# Top 50 most reviewed businesses
top50_most_reviewed <- biz %>% group_by(name) %>% 
  summarise(ReviewCount=sum(review_count, na.rm=T),
            AvgStar=mean(stars, na.rm=T)) %>%
  arrange(desc(ReviewCount)) %>% ungroup() %>%
  top_n(50, ReviewCount)

reviews <- stream_in(file("~/Projects/RScripts/Data/review.json"))
reviews <- fread("~/Projects/RScripts/Data/yelp_review.csv")

tip <- stream_in(file("~/Projects/RScripts/Data/tip.json"))

starbucks <- biz %>% filter(name=="Starbucks") %>% .$business_id %>% unique()
tip %>% filter(business_id %in% starbucks) %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%
  count(word,sort = TRUE) %>%
  ungroup()  %>%
  head(30) %>%
  with(wordcloud(word, n, max.words = 50, colors=brewer.pal(8, "Dark2")))


restaurants_only <- biz %>% filter(grepl("restaurants", tolower(categories)))
tip_restaurants <- tip[tip$business_id %in% unique(restaurants_only$business_id),]

# Tips DataSet ------------------------------------------------------------




