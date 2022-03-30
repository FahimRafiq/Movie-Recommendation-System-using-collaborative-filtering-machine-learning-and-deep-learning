# importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import SlopeOne
from surprise import CoClustering
from surprise import KNNWithMeans

# First we are going to shape our dataframes accordingly for users,movies and ratings

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

ratings = pd.read_csv('u.data', sep='\t', names=r_cols, encoding='latin-1')
print(ratings.head())
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('u.item', sep='|', names=i_cols, encoding='latin-1')
print(movies.head())
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names=u_cols, encoding='latin-1')
print(users.head())

# Assign X as the original ratings dataframe and y as the user_id column of ratings.
X = ratings.copy()
y = ratings['user_id']
# Split into training and test datasets, stratified along user_id
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

df_ratings = X_train.pivot(index='user_id', columns='movie_id', values='rating')
'''Now, our df_ratings dataframe is indexed by user_ids with movie_ids belonging to 
different columns and the values are the ratings with most of the values as Nan 
as each user watches and rates only few movies. Its a sparse dataframe.'''
# print(df_ratings.head())
# Method-01 weighted averaged

df_ratings_dummy = df_ratings.copy().fillna(0)
df_ratings_dummy.head()

# cosine similarity of the ratings
similarity_matrix = cosine_similarity(df_ratings_dummy, df_ratings_dummy)
similarity_matrix_df = pd.DataFrame(similarity_matrix, index=df_ratings.index, columns=df_ratings.index)

print(similarity_matrix_df)


# calculate ratings using weighted sum of cosine similarity
# function to calculate ratings
def calculate_ratings(id_movie, id_user):
    if id_movie in df_ratings:
        cosine_scores = similarity_matrix_df[id_user]  # similarity of id_user with every other user
        ratings_scores = df_ratings[id_movie]  # ratings of every other user for the movie id_movie
        # won't consider users who havent rated id_movie so drop similarity scores and ratings corresponsing to np.nan
        index_not_rated = ratings_scores[ratings_scores.isnull()].index
        ratings_scores = ratings_scores.dropna()
        cosine_scores = cosine_scores.drop(index_not_rated)
        # calculating rating by weighted mean of ratings and cosine scores of the users who have rated the movie
        ratings_movie = np.dot(ratings_scores, cosine_scores) / cosine_scores.sum()
    else:
        return 2.5
    return ratings_movie


print(calculate_ratings(3, 150))  # predicts rating for user_id 150 and movie_id 3


# evaluates on test set
def score_on_test_set():
    user_movie_pairs = zip(X_test['movie_id'], X_test['user_id'])
    predicted_ratings = np.array([calculate_ratings(movie, user) for (movie, user) in user_movie_pairs])
    true_ratings = np.array(X_test['rating'])
    score = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    return score


def recall_weighted():
    user_movie_pairs = zip(X_test['movie_id'], X_test['user_id'])
    predicted_ratings = np.array([calculate_ratings(movie, user) for (movie, user) in user_movie_pairs])
    true_ratings = np.array(X_test['rating'])

    diff_rating = np.abs(predicted_ratings - true_ratings)
    threshold_count = 0
    for rating_weighted in diff_rating:
        if rating_weighted < 0.75:
            threshold_count = threshold_count + 1
    recall_score = threshold_count / predicted_ratings.size
    return recall_score


test_set_score = score_on_test_set()
print("Weighted average RMSD = ", test_set_score)
print("Weighted average recall: ", recall_weighted())

# Approach 2 with ML models
# Define a Reader object
# The Reader object helps in parsing the file or dataframe containing ratings
ratings = ratings.drop(columns='timestamp')
reader = Reader()
# dataset creation
data = Dataset.load_from_df(ratings, reader)

# dataset prep
trainset = data.build_full_trainset()
testset = trainset.build_testset()


def recal_calculate(model):
    model.fit(trainset)
    predictions = model.test(testset)
    true_list = list()
    predicted_list = list()
    for each_user in predictions:
        if each_user[2] >= 3.5:
            true_list.append(each_user[2])
            predicted_list.append((each_user[3]))
    # print(true_list)
    true_a = np.array(true_list)
    predicted_a = np.array(predicted_list)

    difference_a = true_a - predicted_a
    difference_a = np.abs(difference_a)

    difference_list = difference_a.tolist()
    recall_true = 0
    recall_total = 0
    for diff in difference_list:
        if diff < .75:
            recall_true = recall_true + 1
        recall_total = recall_total + 1
    recall = recall_true / recall_total

    return recall


# Define the SVD algorithm object
svd = SVD()
# Evaluate the performance in terms of RMSE
print("SVD RMSD value: ", cross_validate(svd, data, measures=['RMSE'], cv=3))
print("SVD Recall value: ", recal_calculate(svd))

# SlopOnne model
slopeOne = SlopeOne()
print("SlopOne RMSD value: ", cross_validate(slopeOne, data, measures=['RMSE'], cv=3))
print("SlopeOne recall value: ", recal_calculate(slopeOne))

# Coclustering model
# coClustering = CoClustering()
# print(cross_validate(coClustering, data, measures=['RMSE'], cv=3))
# print(recal_calculate(coClustering))

# KnnMeans Model
kNNWithMeans = KNNWithMeans()
print("KNN means RMSD: ", cross_validate(kNNWithMeans, data, measures=['RMSE'], cv=3))
print("KNN means recall: ", recal_calculate(kNNWithMeans))
