import pandas as pd
import sys, os


path = sys.argv[1]

def user_feature(mode):

    if mode == "train":
        data = pd.read_csv( os.path.join(path, 'course','train_preprocess.csv'))
        file_name = "user_feature.csv"
    else :
        data = pd.read_csv(f'data/{mode}_unseen.csv')
        file_name = f"{mode}_unseen_user_feature.csv"

    data_column = data['user_id'].tolist()

    users = pd.read_csv( os.path.join(path,  'users.csv'))
    users = users.fillna("")
    group_id = {}
    for i in range(len(users)):
        interests = users['interests'][i]
        if interests != "":
            for j in interests.split(","):
                if j.split("_")[0] not in group_id:
                    group_id[j.split("_")[0]] = len(group_id)

    users = users.fillna("")
    user_id = {y: x for x, y in users['user_id'].items()}
    
    female, male, other = [], [], []
    tmp =  { x: [] for x in range(len(group_id))}
    for i in range(len(data_column)):
        g = users['gender'][user_id[data['user_id'][i]]]
        F, M, O = 0, 0, 0
        if  g == 'female':
            F = 1
        elif g == 'male':
            M = 1
        elif g == 'other':
            O = 1
        female.append(F)
        male.append(M)
        other.append(O)

        cnt = [0]*len(group_id)
        sub_groups = users['interests'][user_id[data['user_id'][i]]]
        if sub_groups != "":
            for g in sub_groups.split(","):
                    cnt[group_id[g.split("_")[0]]] = 1
        for j in range(len(group_id)):
            tmp[j].append(cnt[j])
    df = pd.DataFrame(list(zip(data_column, female, male, other)), columns =['user_id','female', 'male', 'other'])
    tmp = pd.DataFrame(tmp)
    df = pd.concat( [df, tmp], axis = 1)
    df.to_csv( os.path.join(path, 'course', file_name), index=False)

        
if __name__ == "__main__":
    if not os.path.exists(os.path.join(path, 'course')):
        os.makedirs(os.path.join(path, 'course'))
    user_feature("train")
    user_feature("val")
    user_feature("test")