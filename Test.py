from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
from surprise import accuracy, SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from collections import defaultdict
import numpy as np
import pandas as pd
predicted = defaultdict(list)

# Read only the first million rows of the CSV file
df = pd.read_csv('ratings.csv', nrows=9000000)

# Keep only the columns that we need
df = df[['userId', 'movieId', 'rating']]

# Convert the pandas dataframe to a Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)

# Split the data into training and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train the SVD model on the entire dataset
model = SVD()
model.fit(trainset)

# Get the top-K recommendations for each user in the test set
K = 10
test_user_ids = set([uid for (uid, _, _) in testset])
all_items = set(trainset.all_items())
top_k_recs = defaultdict(list)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at K.
    
    Parameters:
        actual (dict): A dictionary containing the ground truth items for each user.
        predicted (dict): A dictionary containing the predicted items for each user.
        k (int): The number of items to consider for each user.
    
    Returns:
        The mean average precision at K (MAP@K).
    """
    score = 0.0
    for i, uid in enumerate(actual.keys()):
        pred = predicted.get(uid, [])
        act = actual[uid]
        if len(act) > 0 and len(pred) > 0:
            s = 0.0
            num_hits = 0.0
            for j, item in enumerate(pred):
                if item in act and item not in pred[:j]:
                    num_hits += 1.0
                    s += num_hits / (j + 1.0)
            score += s / min(len(act), k)
    return score / len(actual)

def ndcg(actual, predicted, k):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) at rank k.
    """
    def dcg(r):
        """
        Computes the Discounted Cumulative Gain (DCG) for a ranked list of items.
        """
        return np.sum((np.power(2, r) - 1) / np.log2(np.arange(2, r.size + 2)))

    idcg = np.sum([dcg(np.array(actual[u])[:k]) for u in actual])
    dcg = np.sum([dcg(np.array(predicted[u])[:k]) for u in actual])
    return dcg / idcg

for uid in test_user_ids:
    user_items = set(trainset.ur[trainset.to_inner_uid(uid)][0])
    rec_items = list(all_items - user_items)
    if len(rec_items) > 0:
        predictions = model.test([(uid, iid, 1.0) for iid in rec_items])
        top_k_recs[uid] = [(iid, est) for (_, iid, _, est, _) in predictions]
        top_k_recs[uid].sort(key=lambda x: x[1], reverse=True)
        top_k_recs[uid] = top_k_recs[uid][:K]
        predicted[uid] = [iid for (iid, _) in top_k_recs[uid]]

# Calculate MAP and NDCG for the test set
actual = defaultdict(list)
for uid, iid, true_r in testset:
    actual[uid].append(iid)

mapk_score = mapk(actual, predicted, k=K)
ndcg_score = ndcg(actual, predicted, k=K)

print(f"MAP@{K} = {mapk_score:.4f}")
print(f"NDCG@{K} = {ndcg_score:.4f}")
