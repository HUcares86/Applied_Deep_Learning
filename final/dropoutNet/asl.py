import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
# from metric import mapk
import sys
import os
import implicit
import torch
import random


mode = 'val'
number = 50

path = sys.argv[1]

if __name__ == "__main__":

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = pd.read_csv(os.path.join(path, 'course', 'train_preprocess.csv'))
    user_id = {y: x for x, y in data['user_id'].items()}
    data = data.drop(['user_id'], axis=1)
    data, data_column = data.values.tolist(), data.columns.tolist()

    user_items = csr_matrix(np.array(data), dtype=np.float64)
    model = implicit.als.AlternatingLeastSquares(factors=5, iterations=30000)
    model.fit(user_items)
    model.save("asl_model.npz")

    # val = pd.read_csv(os.path.join(path, f'{mode}_seen.csv'))

    # for i in range(len(val)):
    #     user = user_id[val['user_id'][i]]
    #     recs = model.recommend(user, user_items[user], N=number)
    #     pred = []
    #     # if recs[1][0] == 0:
    #     #     pred.extend( [ data_column[recs[0][0]], data_column[recs[0][1]] , data_column[recs[0][2]], data_column[recs[0][3]]] )
    #     # else:
    #     for j in range(number):
    #         pred.append(data_column[recs[0][j]])
    #     val['course_id'][i] = " ".join(str(p) for p in pred)
    # val.to_csv("val.csv", index=False)

    # if mode == 'val':
    #     gt = pd.read_csv(os.path.join(path, f'{mode}_seen.csv'))
    #     gt = gt.fillna("")
    #     gt = gt['course_id'].tolist()
    #     gt = [value.split(' ') for value in gt]
    #     val = val['course_id'].tolist()
    #     val = [value.split(' ') for value in val]
    #     accuracy = mapk(gt, val, k=50)
    #     print(accuracy)
