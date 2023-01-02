# Movie Recommender

The dataset used for this project is available in [kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/code). The main file is an API that provides a recommendation based 
on a movie. 

# Setup 

```
kaggle datasets download -d tmdb/tmdb-movie-metadata
unzip archive.zip
```

# Run 
To run the API, run the following command:
```
    uvicorn main:app --reload
```

# Author 
Bryan Piguave 