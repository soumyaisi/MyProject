"""
This is a recommendation system code. Here we consider 3 types of data and for each type pf data we
have implemented some recommendation engine.
First: This is a small set of movie lens data. From this data we consider two csv files. ratings.csv
has 4 columns - [userId  movieId  rating  timestamp] and movies.csv has 3 columns [movieId title genres].
Second: This is a selected IMDB data and has total 38 columns but we choosed only 5 columns ['Title','Genre','Director','Actors','Plot'].
Third: This is a 1m movielens data and contains some files. users.dat file  has 5 columns ['user_id' 
'gender' 'age' 'occupation' 'zipcode'], movies.dat file has 3 columns ['movie_id' 'title' 'genres'] 
and  ratings.dat has 4 columns ['user_id' 'movie_id' 'rating' 'timestamp'].
"""


#-----------------------------------------First Example-----------------------------------------
"""
Recommendation of simillar movies of a given movie. Using correlation between given movie and other movies.
But this one is not user specific. 
data: https://grouplens.org/datasets/movielens/
"""

#Import libraies.........
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Load the csv file.......
rating = pd.read_csv('ratings_small.csv')
rating.head(2)

movie = pd.read_csv('movies_small.csv')
movie.head(2)

final_data = pd.merge(movie, rating, on='movieId')
final_data.head(2)
print(final_data.columns)


#calculate number of rating per movie and average rating per movie......

avg_rating = pd.DataFrame(final_data.groupby('title',as_index=False)['rating'].mean())
avg_rating.sort_values(["rating"], axis=0,ascending=False, inplace=True) 
avg_rating.head(5)

rating_count = pd.DataFrame(final_data.groupby('title',as_index=False)['rating'].count())
rating_count.sort_values(["rating"], axis=0,ascending=False, inplace=True) 
rating_count.head(5)

avg_rating.columns = ['title', 'rating_avg']
rating_count.columns = ['title', 'rating_count']

rating_avg_cnt = pd.merge(avg_rating, rating_count, on='title')
rating_avg_cnt = rating_avg_cnt.fillna(0)
rating_avg_cnt.head(5)

rating_avg_cnt.sort_values(["rating_count"], axis=0,ascending=False, inplace=True) 
cnt = rating_avg_cnt['rating_count'].tolist()[:10]
names = tuple(rating_avg_cnt['title'].tolist())[:10]
y_pos = np.arange(len(names))
# Create bars and choose color
plt.bar(y_pos, cnt, color = (0.5,0.1,0.5,0.6))
plt.title('top 10 rating count per movie')
plt.xlabel('title')
plt.ylabel('count')
plt.xticks(y_pos, names)
plt.show()

rating_avg_cnt.sort_values(["rating_avg"], axis=0,ascending=False, inplace=True) 
avg = rating_avg_cnt['rating_avg'].tolist()[:10]
names = tuple(rating_avg_cnt['title'].tolist())[:10]
y_pos = np.arange(len(names))
# Create bars and choose color
plt.bar(y_pos, avg, color = (0.5,0.1,0.5,0.6))
plt.title('top 10 rating average per movie')
plt.xlabel('title')
plt.ylabel('average')
plt.xticks(y_pos, names)
plt.show()


#Plot average ratings against the number of ratings...............
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='rating_avg', y='rating_count', data=rating_avg_cnt, alpha=0.4)


#function for caculate correlation of a movie with all users..................
rating_movie_user = final_data.pivot_table(index='userId', columns='title', values='rating')
rating_movie_user = rating_movie_user.fillna(0)
rating_movie_user.head(2)

print(rating_movie_user.index)

def recommended_top_similar(rating_movie_user, rating_avg_cnt, movie_name, reco_cnt, rating_cnt):
    df_movie = rating_movie_user[movie_name]
    similar_movie = rating_movie_user.corrwith(df_movie)
    movie_corr = pd.DataFrame(similar_movie, columns=['corr'])
    movie_corr.dropna(inplace=True)
    movie_corr = movie_corr.reset_index()
    df = pd.merge(movie_corr, rating_avg_cnt, on = 'title')
    df = df[['title','corr', 'rating_count']]
    result = df[df ['rating_count'] > rating_cnt].sort_values('corr', ascending=False).head(reco_cnt)
    return result

movie_name = 'Waiting to Exhale (1995)'
reco_cnt = 3
rating_cnt = 50
result = recommended_top_similar(rating_movie_user, rating_avg_cnt, movie_name, reco_cnt, rating_cnt)
print(result.head())



#-----------------------------------------Second Example-----------------------------------------
"""
This is a content based recommendation system. We calculate cosine similarity between two movies.
"""

#Import libraies.............
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
from nltk.corpus import stopwords 


#Read the data......
data = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')
print(len(data.columns))
data = data[['Title','Genre','Director','Actors','Plot']]
data.columns = ['title','genre','director','actors','plot']
data.head(2)


#Few columns preprocessing..............
data['director'] = data['director'].map(lambda x: x.lower().split(' '))
data['actors'] = data['actors'].map(lambda x: x.lower().split(','))
for index, row in data.iterrows():
    row['actors'] = [x.replace(' ','') for x in row['actors']]
    row['director'] = ''.join(row['director'])
data['genre'] = data['genre'].map(lambda x: x.lower().split(','))


#key word extrction from text data....
data['words'] = ""
for index, row in data.iterrows():
    text = row['plot']
    r = Rake()
    a = r.extract_keywords_from_text(text)
    b = r.get_word_degrees()
    row['words'] = list(b.keys())
    
data.drop(columns = ['plot'], inplace = True)
data.head(2)
print(data.columns)


#prepare a collection of words from these columns............
data.set_index('title', inplace = True)
data['word_collection'] = ''
cols = data.columns
for index, row in data.iterrows():
    words = ''
    for col in cols:
        if col != 'director':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['word_collection'] = words
    
#data.drop(columns = [col for col in data.columns if col!= 'word_collection'], inplace = True)
final_data = data[['word_collection']]
final_data.head(4)


#Use countvectorizer and cosine similarity measure..................
count_vect = CountVectorizer()
word_count = count_vect.fit_transform(data['word_collection'])
similarity_matrix = cosine_similarity(word_count, word_count)
print(similarity_matrix)


#Function for suggesting top movies.............
def top_similar_recommendations(data, movie_name, similarity_matrix, cnt):
    indexs = pd.Series(data.index)
    movies_list = []
    ax = indexs[indexs == movie_name].index[0]
    cosine_vals = pd.Series(similarity_matrix[ax])
    cosine_vals = cosine_vals.sort_values(ascending = False)
    #print(cosine_vals)
    result = list(cosine_vals.iloc[1:cnt].index)    
    for i in result:
        movies_list.append(list(data.index)[i])  
    return movies_list

cnt=11
movie_name = 'The Godfather'
result = top_similar_recommendations(data, movie_name, similarity_matrix, cnt)
print(result)

 
#-----------------------------------------Third Example-----------------------------------------
"""
This is a content based recommendation system. We calculate cosine similarity between two movies.
"""

#...................Initial preprocessing..................
#Import libraies.........
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse.linalg import svds
from keras.layers import Embedding, Reshape, Concatenate
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dot,Flatten,Input, Dense
from keras.models import Model
import math

#Read all files...............
df_rating = pd.read_csv('ratings.dat', sep='::', engine='python', encoding='latin-1',
                    names=['user_id', 'movie_id', 'rating', 'timestamp'])
df_rating.head(2)

df_user = pd.read_csv('users.dat', sep='::', engine='python', encoding='latin-1',
                    names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
df_user.head(2)
df_movie = pd.read_csv('movies.dat', sep='::', engine='python', encoding='latin-1',
                    names=['movie_id', 'title', 'genres'])
df_movie.head(2)

ages = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
occup = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }
df_user['age_categ'] = df_user['age'].apply(lambda x: ages[x])
df_user['occ_categ'] = df_user['occupation'].apply(lambda x: occup[x])
df_user.head(2)

#getting max of user_id and movie_id
max_user_id = df_rating['user_id'].drop_duplicates().max()
max_movie_id = df_rating['movie_id'].drop_duplicates().max()

#............................Exploration Data Analysis.............
sns.set_style('whitegrid')
sns.set(font_scale=1.5)

#Summary statistics of rating
rating_summary = df_rating['rating'].describe()
print(rating_summary)

# Display distribution of rating
sns.distplot(df_rating['rating'].fillna(df_rating['rating'].median()))

# Generate wordcloud for movie title
df_movie['title'] = df_movie['title'].fillna("").astype('str')
movie_title_corpus = ' '.join(df_movie['title'])
wordcloud_title = WordCloud(stopwords=STOPWORDS, background_color='black', height=2000, width=4000).generate(movie_title_corpus)

plt.figure(figsize=(16,8))
plt.imshow(wordcloud_title)
plt.axis('off')
plt.show()

#Merge three dataframes..................
final_data_ = pd.merge(df_movie, df_rating, on='movie_id')
final_data_.head(2)
print(final_data_.columns)
final_data = pd.merge(final_data_, df_user, on='user_id')
final_data.head(2)
#final_data[['title','genres','rating']].sort_values('rating', ascending=False).head(20)
print(final_data.columns)

#Collect all different types of genre
genre_sets = set()
for c in df_movie['genres'].str.split('|').values:
    genre_sets = genre_sets.union(set(c))
print(genre_sets)

#Count of each genre
def genre_count(final_data, genre_sets, col_name):
    count = {}
    for word in genre_sets:
        count[word] = 0
    for word in final_data[col_name].str.split('|'):
        if type(word) == float and pd.isnull(word):
            continue
        for i in [k for k in word if k in genre_sets]:
            if pd.notnull(i):
                count[i] += 1
    count_list = []
    for k,v in count.items():
        count_list.append([k,v])
    count_list.sort(key = lambda x: x[1], reverse = True)
    return count, count_list

count, count_list = genre_count(df_movie, genre_sets, 'genres')
print(count_list[:5])

#Produce genre wordcloud
genres_dict = dict()
subsets = count_list[0:18]
for s in subsets:
    genres_dict[s[0]] = s[1]
print(genres_dict)

genre_wordcloud = WordCloud(width=1000,height=400, background_color='white')
genre_wordcloud.generate_from_frequencies(genres_dict)

f, ax = plt.subplots(figsize=(16, 8))
plt.imshow(genre_wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#.......................Content Based(cosine similarity score of movie genres)................

df_movie['genres'] = df_movie['genres'].str.split('|')
df_movie['genres'] = df_movie['genres'].fillna("").astype('str')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
mat_tfIdf = tf.fit_transform(df_movie['genres'])
print(mat_tfIdf.shape)
print(mat_tfIdf)

cosine_sim = linear_kernel(mat_tfIdf, mat_tfIdf)
print(cosine_sim)

movie_title = df_movie['title']

#Top recommended movie suggestion
def top_movie_reco(df_movie, cosine_sim, movie_title, title, cnt):
    indexes = pd.Series(df_movie.index, index=df_movie['title'])
    ax = indexes[title]
    similarity_val = list(enumerate(cosine_sim[ax]))
    similarity_val = sorted(similarity_val, key=lambda x: x[1], reverse=True)
    similarity_val = similarity_val[1:cnt]
    movie_ids = [i[0] for i in similarity_val]
    score = [i[1] for i in similarity_val]
    res = movie_title.iloc[movie_ids]
    res = res.to_frame()
    res['score'] = score
    return res

title = 'Race the Sun (1996)'
cnt = 10
result = top_movie_reco(df_movie, cosine_sim, movie_title, title, cnt)
print(result)


#........................Collaborative filtering(user similarity and item similarity)....................

df_rating['user_id'] = df_rating['user_id'].fillna(0)
df_rating['movie_id'] = df_rating['movie_id'].fillna(0)
df_rating['rating'] = df_rating['rating'].fillna(df_rating['rating'].mean())

# Randomly sample 1% of the ratings dataset...
sample_data = df_rating.sample(frac=0.01)
print(sample_data.info())

train_data, test_data = cv.train_test_split(sample_data, test_size=0.2)

# Create two user-item matrices, one for training and another for testing.......
train_mat = train_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])
test_mat = test_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])
print(train_mat.shape)
print(test_mat.shape)

# User Similarity Matrix.............
user_correlation = 1 - pairwise_distances(train_mat, metric='correlation')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation.shape)
print(user_correlation[:4, :4])

# Item Similarity Matrix.............
item_correlation = 1 - pairwise_distances(train_mat.T, metric='correlation')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation.shape)
print(item_correlation[:4, :4])

# Function to predict ratings..............
def rating_prediction(df_rating, similarity, type='user'):
    if type == 'user':
        mean_user_rating = df_rating.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (df_rating - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = df_rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

#Some evaluation.............

#Function to calculate RMSE
def rmse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))

user_pred = rating_prediction(train_mat, user_correlation, type='user')
print(user_pred)
item_pred = rating_prediction(train_mat, item_correlation, type='item')
print(item_pred)

print('User-based CF RMSE: ' + str(rmse(user_pred, test_mat)))
print('Item-based CF RMSE: ' + str(rmse(item_pred, test_mat)))

print('User-based CF RMSE: ' + str(rmse(user_pred, train_mat)))
print('Item-based CF RMSE: ' + str(rmse(item_pred, train_mat)))


#...................Model-Based Collaborative Filtering: (SVD)....................

n_users = df_rating.user_id.unique().shape[0]
n_movies = df_rating.movie_id.unique().shape[0]
print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies))

rating_pivot = df_rating.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)
rating_pivot.head()
print(rating_pivot.columns)

R = rating_pivot.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)

sparsity = round(1.0 - len(df_rating) / float(n_users * n_movies), 3)
print ('The sparsity level of MovieLens1M dataset is ' +  str(sparsity * 100) + '%')

U, sigma, Vt = svds(ratings_demeaned, k = 50)
sigma = np.diag(sigma)

#Making Predictions...........
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds = pd.DataFrame(all_user_predicted_ratings, columns = rating_pivot.columns)
preds.head()

def recommend_movies(predictions, userID, movies, original_ratings, num_recommendations):
    user_row_number = userID - 1 # User ID starts at 1, not 0
    sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False)
    user_data = original_ratings[original_ratings.user_id == (userID)]
    user_full = (user_data.merge(movies, how = 'left', left_on = 'movie_id', right_on = 'movie_id').
                     sort_values(['rating'], ascending=False))
    print ('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print ('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies[~movies['movie_id'].isin(user_full['movie_id'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movie_id', right_on = 'movie_id').rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).iloc[:num_recommendations, :-1])
    return user_full, recommendations
    
already_rated, predictions = recommend_movies(preds, 1310, df_movie, df_rating, 20)
# Top 20 movies that User 1310 has rated 
already_rated.head(20)
# Top 20 movies that User 1310 hopefully will enjoy
predictions

#Model Evaluation...............
from surprise import Reader, Dataset, SVD, evaluate
reader = Reader()
data = Dataset.load_from_df(df_rating[['user_id', 'movie_id', 'rating']], reader)
data.split(n_folds=5)
svd = SVD()
evaluate(svd, data, measures=['RMSE'])
trainset = data.build_full_trainset()
svd.train(trainset)
df_rating[df_rating['user_id'] == 1310]
svd.predict(1310, 1994)


#................Deep Learning(Matrix Factorization for Collaborative Filtering).................

df_rating['modified_user_id'] = df_rating['user_id'] - 1
df_rating['modified_movie_id'] = df_rating['movie_id'] - 1
df_movie['modified_movie_id'] = df_movie['movie_id'] - 1
df_user['modified_user_id'] = df_user['user_id'] - 1

shuffled_ratings = df_rating.sample(frac=1., random_state=1234)
Users = shuffled_ratings['modified_user_id'].values
print('Users:', Users, ', shape =', Users.shape)
Movies = shuffled_ratings['modified_movie_id'].values
print('Movies:', Movies, ', shape =', Movies.shape)
Ratings = shuffled_ratings['rating'].values
print('Ratings:', Ratings, ', shape =', Ratings.shape)


K_FACTORS = 100 # The number of dimensional embeddings for movies and users

user_input = Input(shape=[1], name="User-Input")
#user_embedding = Embedding(len(df_user.user_id.unique())+1, 5, name="User-Embedding")(user_input)
user_embedding = Embedding(max_user_id, K_FACTORS, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-users")(user_embedding)
user1 = Dense(100, name='uDense1')(user_vec)
user2 = Dense(100, name='uDense2')(user1)


movie_input = Input(shape=[1], name="Movie-Input")
#movie_embedding = Embedding(len(df_movie.movie_id.unique())+1, 5, name="Movie-Embedding")(movie_input)
movie_embedding = Embedding(max_movie_id, K_FACTORS, name="Movie-Embedding")(movie_input)
movie_vec = Flatten(name="Flatten-movies")(movie_embedding)
movie1 = Dense(100, name='mDense1')(movie_vec)
movie2 = Dense(100, name='mDense2')(movie1)

#prod = Dot(name="Dot-Product", axes=1)([user_vec, movie_vec])
prod = Dot(name="Dot-Product", axes=1)([user2, movie2])
model = Model([user_input, movie_input], prod)

print(model.summary())
model.compile('adam', 'mean_squared_error')

TEST_USER = max_user_id-1 #user_id - 1
TEST_USER = 453-1
TEST_USER = 0
# Function to predict the ratings given User ID and Movie ID
def predict_rating(user_id, movie_id):
    #return trained_model.rate(user_id - 1, movie_id - 1)
    s = model.predict([np.array([user_id]),np.array([movie_id])])
    return s[0][0]

user_ratings = df_rating[df_rating['modified_user_id'] == TEST_USER][['user_id', 'modified_movie_id', 'rating']]

user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(TEST_USER, x['modified_movie_id']), axis=1)
df_movie1 = df_movie[['title','modified_movie_id']]

#result = user_ratings.sort_values(by='rating', ascending=False).merge(df_movie1, on='modified_movie_id', how='inner')
result = user_ratings.sort_values(by='prediction', ascending=False).merge(df_movie1, on='modified_movie_id', how='inner')
result['movie_id'] = result['modified_movie_id'] + 1
result = result.drop(['modified_movie_id'], axis = 1)
print(result.columns)
result.head(15)




