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
    if os.path.exists('data/task6_data.csv'):
        df = po.read_csv('data/task6_data.csv').drop('Unnamed: 0', axis = 1)
            
        return df
    
    else:
        df = po.read_csv('data/task6_train.csv').drop('Unnamed: 0', axis = 1)
        df = po.concat((df, po.read_csv('data/task6_test.csv').drop('Unnamed: 0', axis = 1)), ignore_index=True)
        df.dropna(inplace=True)

        stopwords = list(set(nltk.corpus.stopwords.words("english")))

        ### tokenize & remove funny characters
        df["text"] = df["text"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords))

        df.to_csv('data/task_6_data.csv', index = False)

        return df
