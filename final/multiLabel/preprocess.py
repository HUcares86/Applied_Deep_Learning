import pandas as pd
import sys
import os

path = sys.argv[1]
train = pd.read_csv( os.path.join( path, "train.csv"))
courses = pd.read_csv( os.path.join( path,"courses.csv"))
subgroup = pd.read_csv(  os.path.join( path,"subgroups.csv"))
user = pd.read_csv( os.path.join( path,"users.csv"))
train = train.fillna("")
user = user.fillna("")
courses = courses.fillna("")

group_id = {y: x+1 for x, y in subgroup['subgroup_name'].items()}
user_id = {y: x for x, y in user['user_id'].items()}
course_id = {y: x for x, y in courses['course_id'].items()}

def prerpocess(train):
    output = train[['user_id']].copy()
    output['comment_text'] = ""
    output['list'] = ""
    cnt = 0
    for i in range(len(train)):
        tmp = train['course_id'][i]
        output['comment_text'][i] = user['interests'][user_id[train['user_id'][i]]] + "," + user['recreation_names'][user_id[train['user_id'][i]]] \
                                        + user['occupation_titles'][user_id[train['user_id'][i]]]
        group  = [0]*92
        if tmp != 0:
            for c in tmp.split(" "):
                sub_group = courses['sub_groups'][course_id[c]]
                if sub_group != "":
                    for g in sub_group.split(","):
                        group[group_id[g]] += 1
        if max(group) != 0:
            output['list'][i] = " ".join([ str(j/sum(group)) for j in group])
        else:
            cnt += 1
            output['list'][i] = " ".join([ str(j) for j in group])
    return output

if not os.path.exists(os.path.join(path, 'subgroup')):
    os.makedirs(os.path.join(path, 'subgroup'))
output = prerpocess(train)
output.to_csv( os.path.join( path,'subgroup', 'train_preprocess.csv'), index=False)

val = pd.read_csv( os.path.join( path,"val_seen.csv"))
output = prerpocess(val)
output.to_csv( os.path.join( path, 'subgroup', 'val_preprocess.csv'), index=False)

test = pd.read_csv(  os.path.join( path,"test_seen.csv"))
output = prerpocess(test)
output.to_csv( os.path.join( path, 'subgroup', "test_seen_preprocess.csv"), index=False)

test = pd.read_csv(  os.path.join( path,"test_unseen.csv"))
output = prerpocess(test)
output.to_csv( os.path.join( path, 'subgroup', "test_unseen_preprocess.csv"), index=False)