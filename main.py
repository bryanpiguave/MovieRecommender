from typing import Union
from fastapi import FastAPI
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import preprocessing
from sklearn.metrics.pairwise import sigmoid_kernel
#Content-Based Filtering
from sklearn.feature_extraction.text import TfidfVectorizer


min_max_scaler = preprocessing.MinMaxScaler()

# Getting the data
movies_clean = pd.read_csv("clean_dataset.csv")
movies_ranked = movies_clean.sort_values('weighted_average', ascending=False)
popular = movies_ranked.sort_values('popularity', ascending=False)


# Using Abhishek Thakur's arguments for TF-IDF
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

# Fitting the TF-IDF on the 'overview' text
movies_clean['overview'] = movies_clean['overview'].fillna('')
    
tfv_matrix = tfv.fit_transform(movies_clean['overview'])
movies_scaled = min_max_scaler.fit_transform(movies_clean[['weighted_average', 'popularity']])
movies_norm = pd.DataFrame(movies_scaled, columns=['weighted_average', 'popularity'])
movies_norm.head()

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Reverse mapping of indices and movie titles
indices = pd.Series(movies_clean.index, index=movies_clean['original_title']).drop_duplicates()

def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_clean['original_title'].iloc[movie_indices]


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/item/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    original_title=movies_clean["original_title"].iloc[item_id]
    return {"item_id": item_id, 
            "title":original_title,
            "q": q}


@app.get("/recommendation/{item_id}")
def recomendation_item(item_id: str, q: Union[str, None] = None):
    rec=give_rec('Spy Kids')
    return {"recommendation": rec, "q": q}
