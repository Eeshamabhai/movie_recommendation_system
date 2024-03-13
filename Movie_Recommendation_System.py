# %%
import numpy as np
import pandas as pd

# %%
movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

# %%
credits.head(1)

# %%
movies.head(1)

# %%
movies=movies.merge(credits,on='title')

# %%
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# %%
movies.info()

# %%
movies.head()

# %%
movies.isnull().sum()

# %%
movies.dropna(inplace=True)

# %%
movies.duplicated().sum()

# %%
movies.iloc[0].genres

# %%
import ast

# %%


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# %%
movies['genres']

# %%
movies['genres'].apply(convert)

# %%
movies['genres']=movies['genres'].apply(convert)

# %%
movies.head()

# %%
movies['keywords'].apply(convert)

# %%
movies['keywords']=movies['keywords'].apply(convert)

# %%
movies.head()

# %%
import ast

def convert3(obj):
    L = []
    counter = 0  # Initialize counter
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


# %%
movies['cast'].apply(convert3)

# %%
movies['cast']=movies['cast'].apply(convert3)

# %%
movies['crew'][0]

# %%
#
#import ast

#def fetch_director(obj):
 #   L = []
  #  for i in ast.literal_eval(obj):
     #   if i['job'] == 'Director':
      #      L.append(i['name'])
       #     break
    #return L


# %%
import ast

def fetch_director(obj):
    L = []
    try:
        data = ast.literal_eval(obj)
        if isinstance(data, list):
            for i in data:
                if isinstance(i, dict) and 'job' in i and i['job'] == 'Director':
                    L.append(i['name'])
                    break
    except (ValueError, SyntaxError):
        pass  # Handle the case when obj cannot be evaluated with ast.literal_eval
    
    return L

# %%
movies['crew'].apply(fetch_director)

# %%
movies['crew']=movies['crew'].apply(fetch_director)

# %%
movies.head()

# %%
movies['overview'][0]

# %%
movies['overview'].apply(lambda x:x.split())

# %%
movies['overview']=movies['overview'].apply(lambda x:x.split())

# %%
movies.head()


# %%
movies['genres'].apply(lambda x: [i.replace(" ","")for i in x])

# %%
movies['genres']=movies['genres'].apply(lambda x: [i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x: [i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x: [i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x: [i.replace(" ","")for i in x])



# %%
movies.head()

# %%
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

# %%
movies.head()

# %%
new_df=movies[['movie_id','title','tags']]

# %%
new_df

# %%
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))

# %%
new_df['tags'][0]

# %%
movies.head()

# %%
new_df.head()

# %%
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

# %%
new_df.head()

# %%
import nltk

# %%
!pip install nltk

# %%
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

# %%
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# %%
new_df['tags']=new_df['tags'].apply(stem)

# %%
new_df['tags'] = new_df['tags'].apply(lambda x: stem(x))

# %%
new_df['tags'][0]

# %%
new_df['tags'][1]

# %%
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')

# %%
Vectors=cv.fit_transform(new_df['tags']).toarray()


# %%
Vectors

# %%
Vectors[0]

# %%
cv.get_feature_names()

# %%
stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')

# %%
stem("captain barbossa, long believed to be dead, has come back to life and is headed to the edge of the earth with will turner and elizabeth swann. but nothing is quite as it seems. adventure fantasy action ocean drugabuse exoticisland eastindiatradingcompany loveofone'slife traitor shipwreck strongwoman ship alliance calypso afterlife fighter pirate swashbuckler aftercreditsstinger johnnydepp orlandobloom keiraknightley goreverbinski")

# %%
from sklearn.metrics.pairwise import cosine_similarity

# %%
from sklearn.metrics.pairwise import cosine_similarity
similarities=cosine_similarity(Vectors)

# %%
similarities

# %%
similarities[0]

# %%
new_df[new_df['title']=='Batman Begins'].index[0]

# %%
sorted(list(enumerate(similarities[0])),reverse=True,key=lambda x:x[1])[1:6]

# %%
#def recommend(movie):
 #   movie_index=new_df[new_df['title']==movie].index[0]
  #  distances=similarities[movie_index]
   # movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    #for i in movie_list:
     #   print(new_df.iloc[i[0]].title)

# %%
def recommend(movie):
    # Find the index of the given movie in new_df
    movie_index = new_df[new_df['title'] == movie].index[0]
    
    # Get the cosine similarity distances between the given movie and all other movies
    distances = similarities[movie_index]
    
    # Sort the distances and get the indices of the top 5 most similar movies
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    # Print the titles of the top 5 most similar movies
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# %%
recommend('Avatar')

# %%
recommend('Batman Begins')

# %%
recommend('John Carter')

# %%


# %%



