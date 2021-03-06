import os
import nltk
import pandas as po


### remove stopwords and non-words from tokens list
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        token = token.lower()
        if (token not in stopwords) and (token not in [".",",",";","&","'s", ":", "?", "!","(",")", "@",\
            "'","'m","'no","***","--","...","[","]"]):
            tokens1.append(token)
    return tokens1


def load_and_preprocess():
    df = po.read_csv('data/task6_train.csv').drop('Unnamed: 0', axis = 1)
    df = po.concat((df, po.read_csv('data/task6_test.csv').drop('Unnamed: 0', axis = 1)), ignore_index=True)
    df.dropna(inplace=True)

    stopwords = list(set(nltk.corpus.stopwords.words("english")))

    ### tokenize & remove funny characters
    df["text"] = df["text"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords))

    empty_sent_index = []
    for i, sent in enumerate(df['text']):
        if len(sent) == 0:
            empty_sent_index.append(i)

    df = df.drop(empty_sent_index, axis = 0)
    df = df.reset_index(drop = True)

    df.to_csv('data/task_6_data.csv', index = False)
    
    return df
