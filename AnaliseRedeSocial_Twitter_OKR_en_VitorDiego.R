# Análise de Sentimentos no Twitter

# Configurando o diretório de trabalho
setwd("")
getwd()

## Etapa 1 - Pacotes e Autenticação

# Instalando e Carregando o Pacote twitteR
#install.packages("twitteR")
#install.packages("httr")
#install.packages("wordcloud")

library(twitteR)
library(httr)
library(SnowballC)
library(tm)
library(RColorBrewer)
library(wordcloud)
options(warn=-1)


# Chaves de autenticação no Twitter
key <- ""
secret <- ""
token <- ""
tokensecret <- ""

# Autenticação. Responda 1 quando perguntado sobre utilizar direct connection.
setup_twitter_oauth(key, secret, token, tokensecret)


## Etapa 2 - Conexão e captura dos tweets

# Capturando os tweets em inglês
tema <- "okr"
qtd_tweets <- 1000
lingua <- "en"
tweetdata = searchTwitter(tema, n = qtd_tweets, lang = lingua)

# Visualizando as primeiras linhas do objeto tweetdata
head(tweetdata)


## Etapa 3 - Tratamento dos dados coletados através de text mining

# Funções de limpeza

# Function para limpeza dos tweets
limpaTweets <- function(tweet){
  # Remove http links
  tweet = gsub("(f|ht)(tp)(s?)(://)(.*)[.|/](.*)", " ", tweet)
  tweet = gsub("http\\w+", "", tweet)
  # Remove retweets
  tweet = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", " ", tweet)
  # Remove “#Hashtag”
  tweet = gsub("#\\w+", " ", tweet)
  # Remove nomes de usuarios “@people”
  tweet = gsub("@\\w+", " ", tweet)
  # Remove pontuacão
  tweet = gsub("[[:punct:]]", " ", tweet)
  # Remove os números
  tweet = gsub("[[:digit:]]", " ", tweet)
  # Remove espacos desnecessários
  tweet = gsub("[ \t]{2,}", " ", tweet)
  tweet = gsub("^\\s+|\\s+$", "", tweet)
  # Convertendo encoding de caracteres e convertendo para letra minúscula
  tweet <- stringi::stri_trans_general(tweet, "latin-ascii")
  tweet <- tryTolower(tweet)
  tweet <- iconv(tweet, from = "UTF-8", to = "ASCII")
}

# Function para limpeza de Corpus
limpaCorpus <- function(myCorpus){
  library(tm)
  myCorpus <- tm_map(myCorpus, tolower)
  # Remove pontuação
  myCorpus <- tm_map(myCorpus, removePunctuation)
  # Remove números
  myCorpus <- tm_map(myCorpus, removeNumbers)
}

# Converte para minúsculo
tryTolower = function(x)
{
  # Cria um dado missing (NA)
  y = NA
  # faz o tratamento do erro
  try_error = tryCatch(tolower(x), error=function(e) e)
  # se não der erro, transforma em minúsculo
  if (!inherits(try_error, "error"))
    y = tolower(x)
  # Retorna o resultado
  return(y)
}

# Tratamento (limpeza, organização e transformação) dos dados coletados
tweetlist <- sapply(tweetdata, function(x) x$getText())
tweetlist <- iconv(tweetlist, to = "utf-8", sub="")
tweetlist <- limpaTweets(tweetlist)
tweetcorpus <- Corpus(VectorSource(tweetlist))
tweetcorpus <- tm_map(tweetcorpus, removePunctuation)
tweetcorpus <- tm_map(tweetcorpus, content_transformer(tolower))
tweetcorpus <- tm_map(tweetcorpus, function(x)removeWords(x, stopwords()))

# Convertendo o objeto Corpus para texto plano
# tweetcorpus <- tm_map(tweetcorpus, PlainTextDocument)
termo_por_documento = as.matrix(TermDocumentMatrix(tweetcorpus), control = list(stopwords = c(stopwords("english"))))


## Etapa 4 - Wordcloud, associação entre as palavras e dendograma

# Gerando uma nuvem palavras
pal2 <- brewer.pal(8,"Dark2")

wordcloud(tweetcorpus, 
          min.freq = 1, 
          scale = c(3.5,0.25), 
          random.color = F, 
          max.word = 100, 
          random.order = F,
          rot.per = 0.35,
          colors = pal2)

# Convertendo o objeto texto para o formato de matriz
tweettdm <- TermDocumentMatrix(tweetcorpus)
tweettdm

# Encontrando as palavras que aparecem com mais frequência
findFreqTerms(tweettdm, lowfreq = 10)


# Buscando associações
findAssocs(tweettdm, 'result', 0.60)

# Removendo termos esparsos (não utilizados frequentemente)
tweet2tdm <- removeSparseTerms(tweettdm, sparse = 0.9)

# Criando escala nos dados
tweet2tdmscale <- scale(tweet2tdm)

# Distance Matrix
tweetdist <- dist(tweet2tdmscale, method = "euclidean")

# Preparando o dendograma - agrupamento de palavras
tweetfit <- hclust(tweetdist)

# Criando o dendograma (verificando como as palavras se agrupam)
plot(tweetfit)
typeof(tweetfit)
View(tweetfit)

# Verificando os grupos
cutree(tweetfit, k = 2)

# Visualizando os grupos de palavras no dendograma
rect.hclust(tweetfit, k = 2, border = "red")


## Usando Classificador Naive Bayes para analise de sentimento
# https://cran.r-project.org/src/contrib/Archive/Rstem/
# https://cran.r-project.org/src/contrib/Archive/sentiment/

#install.packages("Rstem_0.4-1.tar.gz", sep = "", repos = NULL, type = "source")
#install.packages("sentiment_0.2.tar.gz",sep = "", repos = NULL, type = "source")
#install.packages("ggplot2")
library(Rstem)
library(sentiment)
library(ggplot2)

# Classificando emocao
class_emo = classify_emotion(tweetlist, algorithm = "bayes", prior = 1.0)
emotion = class_emo[,7]

# Substituindo NA's por "Neutro"
emotion[is.na(emotion)] = "Neutro"

# Classificando polaridade
class_pol = classify_polarity(tweetlist, algorithm = "bayes")
polarity = class_pol[,4]

# Gerando um dataframe com o resultado
sent_df = data.frame(text = tweetlist, emotion = emotion,
                     polarity = polarity, stringsAsFactors = FALSE)
View(sent_df)

# Ordenando o dataframe
sent_df = within(sent_df,
                 emotion <- factor(emotion, levels = names(sort(table(emotion), 
                                                                decreasing=TRUE))))

# Emoções encontradas
ggplot(sent_df, aes(x = emotion)) +
  geom_bar(aes(y = ..count.., fill = emotion)) +
  scale_fill_brewer(palette = "Dark2") +
  labs(x = "Categorias", y = "Numero de Tweets") 

# Polaridade
ggplot(sent_df, aes(x=polarity)) +
  geom_bar(aes(y=..count.., fill=polarity)) +
  scale_fill_brewer(palette="RdGy") +
  labs(x = "Categorias de Sentimento", y = "Numero de Tweets")


