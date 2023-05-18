

## 1. Basics of Recommendation Algorithm
"""

from scipy.spatial.distance import cosine
import sklearn.metrics as metrics
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error

#TODO: load dataset into variable M
import numpy as np
import pandas as pd
M = np.array([[4, 3, 2, 3],
            [1, 2, 3, 1],
            [np.nan, 2, 1, np.nan],
            [4, 3, np.nan, np.nan]])
pd.DataFrame(M)

"""### Compute similarities

#### Cosine
"""

import math
def cosine_similarity(v1,v2, metric='cosine'):
    #metric: cosine or correlation
    if metric == 'correlation':
        v1 = v1 - np.nanmean(v1)
        v2 = v2 - np.nanmean(v2)
    "compute similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        if np.isnan(x) or np.isnan(y): continue
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def sim_matrix(M, dimension='user', metric='cosine'):
    N = M.shape[0] if dimension == 'user' else M.shape[1]
    sim = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            if i == j:
                sim[i,j] = 0 #Cancel out the effect of self-similarity in the sums later
                continue
            if dimension == 'user':
                v1, v2 = M[i,:], M[j,:]
            else:
                v1, v2 = M[:,i], M[:,j]
            sim[i][j] = cosine_similarity(v1,v2,metric)
    return sim

cosine_similarity(M[0,:], M[2,:], 'cosine')

sim_matrix(M, 'user')

sim_matrix(M, 'item')

"""#### Pearson"""

cosine_similarity(M[0,:], M[2,:], 'correlation')

sim_matrix(M, 'user', 'correlation')

sim_matrix(M, 'item', 'correlation')

"""### a) Compute the missing rating in this table using user-based collaborative filtering (CF). (Use cosine similarity, then use Pearson similarity). Assume taking all neighbors"""

def user_cf(M, metric='cosine'):
    pred = np.copy(M)
    n_users, n_items = M.shape
    avg_ratings = np.nanmean(M, axis=1)
    sim_users = sim_matrix(M, 'user', metric)
    for i in range(n_users):
        for j in range(n_items):
            if np.isnan(M[i,j]):
                pred[i,j] = #TODO: finish the calculation here
    return pred

print("User-based CF (Cosine): \n" + str(pd.DataFrame(user_cf(M, 'cosine'))))
print("User-based CF (Pearson): \n" + str(pd.DataFrame(user_cf(M, 'correlation'))))

"""### b) Similarly, computing the missing rating using item-based CF."""

def item_cf(M, metric='cosine'):
    pred = np.copy(M)
    n_users, n_items = M.shape
    avg_ratings = np.nanmean(M, axis=0)
    sim_items = sim_matrix(M, 'item', metric)
    for i in range(n_users):
        for j in range(n_items):
            if np.isnan(M[i,j]):
                pred[i,j] = #TODO: finish the calculation here
    return pred

print("Item-based CF (Cosine): \n" + str(pd.DataFrame(item_cf(M, 'cosine'))))
print("Item-based CF (Pearson): \n" + str(pd.DataFrame(item_cf(M, 'correlation'))))

"""## 2. Evaluating Recommendation Algorithms

### Predictive Accuracy
"""

M_result = np.asarray([[4,3,2,3], 
                [1,2,3,1],
                [1,2,1,2],
                [4,3,2,4]])
pd.DataFrame(M_result)

def evaluateRS(ratings, groundtruth, method='user_cf', metric='cosine'):
    #method: user_cf and item_cf, metric: cosine and correlation
    if method == 'user_cf':
        prediction = user_cf(ratings, metric)
    else:
        prediction = item_cf(ratings, metric)
    MSE = mean_squared_error(prediction, groundtruth)
    RMSE = round(sqrt(MSE),3)
    print("RMSE using {0} approach ({2}) is: {1}".format(method, RMSE, metric))
    print(pd.DataFrame(prediction))
    return

#TODO: evaluate the predictive accuracy 
evaluteRS(M, M_result, 'user_cf', 'cosine')
evaluteRS(M, M_result, 'user_cf', 'correlation')
evaluteRS(M, M_result, 'item_cf ', 'cosine')
evaluteRS(M, M_result, 'item_cf ', 'correlation')

"""### Ranking Accuracy"""

import scipy.stats as stats

def evaluate_rank(ratings, groundtruth, method='user_cf', metric='cosine'):
    #metric: cosine vs correlation
    if method == 'user_cf':
        prediction = user_cf(ratings, metric)
    else:
        prediction = item_cf(ratings, metric)
    
    avg_tau = 0
    for i in range(n_users):
        tau, p_value = stats.kendalltau(M_result[i,:], prediction[i,:])
        avg_tau += tau
    avg_tau = avg_tau / n_users
    clear_output(wait=True)
    return avg_tau


#TODO: calculate the ranking accuracy
results = []
for method in ['user_cf', 'item_cf']: 
    for metric in ['cosine ', 'correlation']:
        rank_acc = evaluate_rank(M, M_result, method, metric)
        results += ["Rank accuracy of {0} with {1} metric: {2}".format(method[1], metric, rank_acc)]

print("\n".join(results))

"""# IV. Exercises

1. Classification
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split , KFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# import iris dataset
iris = datasets.load_iris()
# np.c_ is the numpy concatenate function
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                      columns= iris['feature_names'] + ['target'])
iris_df.head()

x= iris_df.iloc[:, :-1]
y= iris_df.iloc[:, -1]

# split the data into train and test sets
x_train, x_test, y_train, y_test= train_test_split(x, y,
                                                   test_size= 0.2,
                                                   shuffle= True, #shuffle the data to avoid bias
                                                   random_state= 0)
x_train= np.asarray(x_train)
y_train= np.asarray(y_train)

x_test= np.asarray(x_test)
y_test= np.asarray(y_test)

print(f'training set size: {x_train.shape[0]} samples \ntest set size: {x_test.shape[0]} samples')

scaler= Normalizer().fit(x_train) # the scaler is fitted to the training set
normalized_x_train= scaler.transform(x_train) # the scaler is applied to the training set
normalized_x_test= scaler.transform(x_test) # the scaler is applied to the test set

print('x train before Normalization')
print(x_train[0:5])
print('\nx train after Normalization')
print(normalized_x_train[0:5])

## Before
# View the relationships between variables; color code by species type
di= {0.0: 'Setosa', 1.0: 'Versicolor', 2.0:'Virginica'} # dictionary

before= sns.pairplot(iris_df.replace({'target': di}), hue= 'target')
before.fig.suptitle('Pair Plot of the dataset Before normalization', y=1.08)

## After
iris_df_2= pd.DataFrame(data= np.c_[normalized_x_train, y_train],
                        columns= iris['feature_names'] + ['target'])
di= {0.0: 'Setosa', 1.0: 'Versicolor', 2.0: 'Virginica'}
after= sns.pairplot(iris_df_2.replace({'target':di}), hue= 'target')
after.fig.suptitle('Pair Plot of the dataset After normalization', y=1.08)

def distance_ecu(x_train, x_test_point):
  """
  Input:
    - x_train: corresponding to the training data
    - x_test_point: corresponding to the test point

  Output:
    -distances: The distances between the test point and each point in the training data.

  """
  distances= []  ## create empty list called distances
  for row in range(len(x_train)): ## Loop over the rows of x_train
      current_train_point= x_train[row] #Get them point by point
      current_distance= 0 ## initialize the distance by zero

      for col in range(len(current_train_point)): ## Loop over the columns of the row
          
          current_distance += (current_train_point[col] - x_test_point[col]) **2
          ## Or current_distance = current_distance + (x_train[i] - x_test_point[i])**2
      current_distance= np.sqrt(current_distance)

      distances.append(current_distance) ## Append the distances

  # Store distances in a dataframe
  distances= pd.DataFrame(data=distances,columns=['dist'])
  return distances

def nearest_neighbors(distance_point, K):
    """
    Input:
        -distance_point: the distances between the test point and each point in the training data.
        -K             : the number of neighbors

    Output:
        -df_nearest: the nearest K neighbors between the test point and the training data.

    """

    # Sort values using the sort_values function
    df_nearest= distance_point.sort_values(by=['dist'], axis=0)

    ## Take only the first K neighbors
    df_nearest= df_nearest[:K]
    return df_nearest

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=50)

print ("Training size: %d" %len(y_train))
print ("Test size    : %d" %len(y_test))

clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print ("Print results for 20 test data points:")
print ("Predicted labels: ", y_pred[20:40])
print ("Ground truth    : ", y_test[20:40])

from sklearn.metrics import accuracy_score
print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print ("Accuracy of 10NN with major voting: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

def myweight(distances):
    sigma2 = .5 # we can change this number
    return np.exp(-distances**2/sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print ("Accuracy of 10NN (customized weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))

"""2. Recommendation Systems"""

import pandas as pd
import numpy as np

# Read the movies.csv file
movies_df = pd.read_csv('movies.csv', encoding="ISO-8859-1")
# Read the users.csv file
users_df = pd.read_csv('users.csv', encoding="ISO-8859-1")
# Read the ratings.csv file
ratings_df = pd.read_csv('ratings.csv', encoding="ISO-8859-1")

# Find list of used genres which is used to category the movies.
movies_df['genres'] = movies_df['genres'].fillna('')

unique_genres = []
for genres in movies_df['genres'].str.split('|'):
    for genre in genres:
        if genre not in unique_genres:
            if genre != "":
                unique_genres.append(genre)
print(unique_genres)

Ij = np.zeros((movies_df.shape[0], len(unique_genres)), dtype=int)
for i, genres in enumerate(movies_df['genres'].str.split('|')):
    for genre in genres:
        if (not isinstance(genre, str)) or (genre == '') or (genre not in unique_genres):
            continue
        j = unique_genres.index(genre)
        Ij[i, j] = 1

np.set_printoptions(formatter={'all': lambda x: f"{x}, "})
print(Ij[:4])

# Vectorize the relationship between users and genres and put them into Uj (if user rate for a movie,
# he/she has the related history with the movies’genres).

merged_df = pd.merge(ratings_df, movies_df, on='movie_id', how='left')

# Get the unique genres
unique_genres = set('|'.join(merged_df['genres'].tolist()).split('|'))

# Create a dictionary to map each genre to its index in the unique genres list
genre_index_dict = {genre: i for i, genre in enumerate(unique_genres)}

num_users = ratings_df['user_id'].nunique()
Uj = np.zeros((num_users, len(unique_genres)), dtype=int)
for index, row in ratings_df.iterrows():
    user_id = row['user_id']
    movie_id = row['movie_id']
    rating = row['rating']
    
    # Get the 'genres' column using the 'movieId' from the merged dataset
    genres = merged_df.loc[merged_df['movie_id'] == movie_id]['genres'].tolist()[0].split('|')
    
    # For each genre in the 'genres' column, set the corresponding entry in Uj to 1
    for genre in genres:
        j = genre_index_dict[genre]
        Uj[user_id-1, j] = 1
Uj = Uj[:, :-1]
print(Uj[:4])

# Compute the cosine_similarity between movies and users. Hint: you can use
# sklearn.metrics.pairwise and cosine_similarity for quick calculation.

from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity(Uj, Ij)


print(cos_sim)

#Collaborative Filtering Recommendation Model by Users

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np

# Load the data
ratings = pd.read_csv('ratings.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.5)

# Create user-item matrices for training and testing data
train_data_matrix = train_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
test_data_matrix = test_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Calculate user similarity
user_correlation = 1 - pairwise_distances(train_data_matrix, metric='cosine')

# Replace NaN values with 0 and set diagonal values to 0
np.fill_diagonal(user_correlation, 0)
user_correlation[np.isnan(user_correlation)] = 0

print(user_correlation)

from sklearn.model_selection import train_test_split

# Split the data into training and test sets
train_data, test_data = train_test_split(ratings_df, test_size=0.5)

# Create matrix for users, movies, and ratings in the training dataset
train_data_matrix = train_data.pivot_table(index='user_id', columns='movie_id', values='rating').astype('float64').fillna(0)

# Create matrix for users, movies, and ratings in the test dataset
test_data_matrix = test_data.pivot_table(index='user_id', columns='movie_id', values='rating').astype('float64').fillna(0)

# Calculate the user correlation
from scipy.spatial.distance import correlation

def calculate_user_correlation(train_data_matrix):
    # Calculate the correlation between users using the 'correlation' function from scipy
    user_correlation = 1 - pairwise_distances(train_data_matrix.values, metric='correlation')
    np.fill_diagonal(user_correlation, 0) # Set the diagonal values to 0 to avoid recommending the same item
    return user_correlation

user_correlation = calculate_user_correlation(train_data_matrix)

print(user_correlation)

def predict_user_based(user_correlation, train_data_matrix, user_id, item_id, k=10):
    # Calculate the mean rating for each user in the training dataset
    mean_user_rating = train_data_matrix.mean(axis=1)

    # Calculate the similarity between the active user and all other users
    sim_scores = user_correlation[user_id-1]

    # Select the top k most similar users
    top_similar_users = sim_scores.argsort()[::-1][1:k+1]

    # Calculate the weighted average of the ratings given by the k neighbors to the item
    item_ratings = train_data_matrix.loc[:, item_id]
    item_ratings = item_ratings[top_similar_users]
    sim_scores = sim_scores[top_similar_users]
    predicted_rating = np.dot(item_ratings, sim_scores) / sim_scores.sum()

    # Normalize the predicted rating by adding the mean rating of the active user
    predicted_rating += mean_user_rating[user_id]
    return predicted_rating

from sklearn.model_selection import train_test_split
from scipy.spatial.distance import correlation
from sklearn.metrics import mean_squared_error
import numpy as np

# Split the data into training and test sets
train_data, test_data = train_test_split(ratings_df, test_size=0.5)

# Create matrix for users, movies, and ratings in the training dataset
train_data_matrix = train_data.pivot_table(index='user_id', columns='movie_id', values='rating').astype('float64')

# Create matrix for users, movies, and ratings in the test dataset
test_data_matrix = test_data.pivot_table(index='user_id', columns='movie_id', values='rating').astype('float64')

# Calculate the user correlation
def calculate_user_correlation(train_data_matrix):
    # Calculate the correlation between users using the 'correlation' function from scipy
    user_correlation = 1 - pairwise_distances(train_data_matrix.values, metric='correlation')
    np.fill_diagonal(user_correlation, 0) # Set the diagonal values to 0 to avoid recommending the same item
    return user_correlation

user_correlation = calculate_user_correlation(train_data_matrix)

# Function to predict ratings based on user similarity
def predict_user_rating(train_data_matrix, user_correlation):
    # Calculate the weighted average of the ratings of similar users
    user_similarity_sum = np.dot(user_correlation, train_data_matrix)
    similarity_sum = np.abs(user_correlation).sum(axis=1).reshape(-1, 1)
    user_predicted_ratings = user_similarity_sum / similarity_sum
    
    return user_predicted_ratings

# Predict on train dataset and compare the RMSE with the test dataset
train_predicted_ratings = predict_user_rating(train_data_matrix, user_correlation)
test_predicted_ratings = predict_user_rating(test_data_matrix, user_correlation)

train_rmse = mean_squared_error(train_predicted_ratings, train_data_matrix, squared=False)
test_rmse = mean_squared_error(test_predicted_ratings, test_data_matrix, squared=False)

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Split data into train and test sets
train_data, test_data = train_test_split(ratings_df, test_size=0.5)

# Create matrix for users, movies and ratings in both training and testing datasets
train_data_matrix = train_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
test_data_matrix = test_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Calculate the user correlation
def pearson_similarity(u1, u2):
    # Find the common movies rated by both users
    common_movies = [movie_id for movie_id in train_data_matrix[u1].index if movie_id in train_data_matrix[u2].index]
    
    # If there are no common movies, return 0
    if len(common_movies) == 0:
        return 0
    
    # Calculate the means for both users
    u1_mean = train_data_matrix.loc[u1, common_movies].mean()
    u2_mean = train_data_matrix.loc[u2, common_movies].mean()
    
    # Calculate the numerator and denominator of the correlation coefficient
    numerator = sum((train_data_matrix.loc[u1, common_movies] - u1_mean) * (train_data_matrix.loc[u2, common_movies] - u2_mean))
    denominator = sqrt(sum((train_data_matrix.loc[u1, common_movies] - u1_mean) ** 2) * sum((train_data_matrix.loc[u2, common_movies] - u2_mean) ** 2))
    
    # If the denominator is 0, return 0
    if denominator == 0:
        return 0
    
    # Return the correlation coefficient
    return numerator / denominator

# Implement a predict based on user correlation coefficient
def predict_user(user_id, movie_id):
    # Find the users who have rated the movie
    users_who_rated = train_data_matrix[movie_id].dropna().index
    
    # Calculate the similarities between the user and all users who have rated the movie
    similarities = [pearson_similarity(user_id, other_user) for other_user in users_who_rated]
    
    # Get the ratings and similarities for the users who have rated the movie
    ratings = train_data_matrix.loc[users_who_rated, movie_id]
    similarities = pd.Series(similarities, index=users_who_rated)
    
    # Calculate the weighted mean of the ratings based on the similarities
    weighted_mean = (ratings * similarities).sum() / similarities.sum()
    
    return weighted_mean

# Predict on train dataset and compare the RMSE with the test dataset
train_preds = []
test_preds = []

for index, row in train_data.iterrows():
    user_id = row['user_id']
    movie_id = row['movie_id']
    train_pred = predict_user(user_id, movie_id)
    train_preds.append(train_pred)
    
for index, row in test_data.iterrows():
    user_id = row['user_id']
    movie_id = row['movie_id']
    test_pred = predict_user(user_id, movie_id)
    test_preds.append(test_pred)
    
train_rmse = sqrt(mean_squared_error(train_data['rating'], train_preds))
test_rmse = sqrt(mean_squared_error(test_data['rating'], test_preds))

print("Train RMSE: ", train_rmse)
print("Test RMSE: ", test_rmse)

# Split the dataset into train and test sets
train_data, test_data = train_test_split(ratings_df, test_size=0.5)

# Create rating matrix for users, movies, and ratings in both training and testing datasets
train_data_matrix = train_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
test_data_matrix = test_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Calculate the item-item similarity matrix
item_correlation = train_data_matrix.corr(method='pearson', min_periods=10)

# Define function to predict ratings based on item similarity
def predict_item_similarity(ratings, similarity):
    # Normalizing the ratings by subtracting the average rating of each user
    ratings_diff = ratings.sub(ratings.mean(axis=1), axis=0)
    
    # Make predictions by multiplying the user's rating vector with the item-item similarity matrix
    pred = similarity.dot(ratings_diff).div(similarity.sum(axis=1))
    
    # Add back the average rating of each user
    pred += ratings.mean(axis=1)
    return pred

# Predict on train dataset and evaluate RMSE
train_item_pred = predict_item_similarity(train_data_matrix, item_correlation)
train_rmse = rmse(train_item_pred, train_data_matrix.values)
print(f'Train RMSE: {train_rmse}')

# Predict on test dataset and evaluate RMSE
test_item_pred = predict_item_similarity(test_data_matrix, item_correlation)
test_rmse = rmse(test_item_pred, test_data_matrix.values)
print(f'Test RMSE: {test_rmse}')
----------


# I. Classification
"""

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

digits = datasets.load_digits()

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
        random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
    
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets, transforms
mnist = datasets.MNIST(root='', train=True, download=True)

print("Number of training example: ", mnist.train_data.shape)
print("Image information ", mnist[0])

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot  as plt 
# %matplotlib inline
plt.imshow(mnist[0][0])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fully = nn.Sequential(
            nn.Linear(28*28, 10)
        )
        
    def forward(self, x):
        x = x.view([-1,28*28])
        x = self.fully(x)
        x = F.log_softmax(x, dim=1)
        return x

train_loader = torch.utils.data.DataLoader(datasets.MNIST(root=".", train=True, transform=transforms.Compose([transforms.ToTensor()])), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(root=".", train=False, transform=transforms.Compose([transforms.ToTensor()])), batch_size=1, shuffle=True)

def train():
    learning_rate = 1e-3
    num_epochs = 3
    
    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = net(data)

            loss = F.nll_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print('Epoch = %f. Batch = %s. Loss = %s' % (epoch, batch_idx, loss.item()))  
                
    return net

net = train()

net.eval()
test_loss = 0
correct = 0
total = 0

for data, target in test_loader:
    total += len(target)
    output = net(data)
    pred = output.max(1, keepdim=True)[1]
    correct += target.eq(pred.view_as(target)).sum()
    
print("Correct out of %s" % total, correct.item())
print("Percentage accuracy", correct.item()*100/10000.)

"""# II. Linear Regression"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('gapminder.csv')

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

X_fertility = df['fertility'].values.reshape(-1, 1)
y = df['life'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_fertility, y, test_size=0.3, random_state=42)

# Create th prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1, 1)

# Fit the model to the data
reg.fit(X_train, y_train)

# compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print $R^2$
print(reg.score(X_fertility, y))

# Plot regression line on scatter plot
sns.scatterplot(x='fertility', y='life', data=df)
plt.plot(prediction_space, y_pred, color='black', linewidth=3)

features = pd.read_csv('gapminder.csv')
df = pd.read_csv('gapminder.csv')
del features['life']
del features['Region']

y_life = df['life'].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(features, y_life, test_size=0.3, random_state=42)

reg_all = LinearRegression()
reg_all.fit(x_train, y_train)
print(reg_all.score(features, y_life))









"""# Linear Regression using PyTorch"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

N = 10 # number of data points
m = .9
c = 1
x = np.linspace(0,2*np.pi,N)
y = m*x + c + np.random.normal(0,.3,x.shape)
plt.figure()
plt.plot(x,y,'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D data (#data = %d)' % N)
plt.show()

import torch

"""## Dataset"""

from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        sample = {
            'feature': torch.tensor([1,self.x[idx]]), 
            'label': torch.tensor([self.y[idx]])}
        return sample

dataset = MyDataset(x, y)
for i in range(len(dataset)):
    sample = dataset[i]
    print(i, sample['feature'], sample['label'])

"""## Dataloader"""

from torch.utils.data import DataLoader

dataset = MyDataset(x, y)
batch_size = 4
shuffle = True
num_workers = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

import pprint as pp
for i_batch, samples in enumerate(dataloader):
    print('\nbatch# = %s' % i_batch)
    print('samples: ')
    pp.pprint(samples)

"""## Model"""

import torch.nn as nn
import torch.nn.functional as F
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out

"""### Setting a model for our problem"""

input_dim = 2
output_dim = 1

model = MyModel(input_dim, output_dim)

"""## Cost function

Often called loss or error
"""

cost = nn.MSELoss()

"""## Minimizing the cost function

In other words training (or learning from data)
"""

num_epochs = 10
l_rate = 0.01
optimiser = torch.optim.SGD(model.parameters(), lr = l_rate) 

dataset = MyDataset(x, y)
batch_size = 4
shuffle = True
num_workers = 4
training_sample_generator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

for epoch in range(num_epochs):
    print('Epoch = %s' % epoch)
    for batch_i, samples in enumerate(training_sample_generator):
        predictions = model(samples['feature'])
        error = cost(predictions, samples['label'])
        print('\tBatch = %s, Error = %s' % (batch_i, error.item()))
        optimiser.zero_grad()
        error.backward()
        optimiser.step()

"""## Lets see how well the model has learnt the data"""

x_for_plotting = np.linspace(0, 2*np.pi, 1000)
design_matrix = torch.tensor(np.vstack([np.ones(x_for_plotting.shape), x_for_plotting]).T, dtype=torch.float32)
print('Design matrix shape:', design_matrix.shape)

y_for_plotting = model.forward(design_matrix)
print('y_for_plotting shape:', y_for_plotting.shape)

plt.figure()
plt.plot(x,y,'o')
plt.plot(x_for_plotting, y_for_plotting.data.numpy(), 'r-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D data (#data = %d)' % N)
plt.show()





