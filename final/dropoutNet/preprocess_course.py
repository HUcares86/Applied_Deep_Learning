import pandas as pd
import numpy as np
from datetime import datetime
import sys, os

path = sys.argv[1]

subgroup = pd.read_csv( os.path.join( path, "subgroups.csv"))
group_id = {y: x+1 for x, y in subgroup['subgroup_name'].items()}
threshold = 720

def data_preprocess(in_data):
    in_data = in_data.fillna("")
    data =  in_data[['user_id']].copy()
    
    for i in range(len(in_data)):
        for c in in_data['course_id'][i].split(" "):
            if c not in data.columns:
                data[c] = 0
            data[c][i] = 1
    # data = data.drop(['user_id'], axis=1)
    # return data.values.tolist(), data.columns.tolist()
    return data

def add_feature(i, course, course_id, data_column, date, price, days, tmp):

    sub_groups = course['sub_groups'][course_id[data_column[i]]] 
    cnt = [0]*92
    if sub_groups != "":
        for g in sub_groups.split(","):
            cnt[group_id[g]] = 1
    for j in range(1,92):
        tmp[j].append(cnt[j])

    date = datetime(2021, 11, 15) - date
    # if price[i] == 1:
        # print( min(date.days / threshold, 1), course['course_name'][course_id[data_column[i]]] )
    if course['course_price'][course_id[data_column[i]]] == 0:
        price.append(1)
        date = abs(date.days)/threshold
    else:
        price.append(0)
        date = 1
    days.append(date)

    return price, days, tmp

def course_feature(data, mode):

    course = pd.read_csv( os.path.join( path,'courses.csv') )
    course = course.fillna("")
    if mode != "new":
        data_column = data.drop(['user_id'], axis=1).columns.tolist()
    else:
        data_column = course['course_id'].tolist()
    course_id = {y: x for x, y in course['course_id'].items()}
    new_course_id, price, days = [], [], []
    tmp =  { x: [] for x in range(1,92)}

    for i in range(len(data_column)):
        
        date = course['course_published_at_local'][course_id[data_column[i]]]
        format ='%Y-%m-%d'
        if date == "":
            continue

        date = datetime.strptime(date.split(" ")[0], format)
        if mode == 'new':
            if  date < datetime(2021, 12, 15) and datetime(2021, 10, 1)  < date :
                new_course_id.append(course['course_id'][course_id[data_column[i]]] )
                price, days, tmp = add_feature(i, course, course_id, data_column, date, price, days, tmp)

        else:
            new_course_id.append(course['course_id'][course_id[data_column[i]]] )
            price, days, tmp = add_feature(i, course, course_id, data_column, date, price, days, tmp)
    # days = [float(i)/threshold for i in days]
    df = pd.DataFrame({
        'course_id': new_course_id,
        'price': price,
        'days': days
    })
    tmp = pd.DataFrame(tmp)
    # df = pd.concat( [df, tmp], axis = 1)
    return df
    
def test_uid(data, val):
    user_id = {y:x for x, y in data['user_id'].items()}
    val['uid'] = 0
    for i in range(len(val)):
        val['uid'][i] = user_id[val['user_id'][i]]
    return val


if __name__ == "__main__":

    if not os.path.exists(os.path.join(path, 'course')):
        os.makedirs(os.path.join(path, 'course'))

    data = pd.read_csv(os.path.join( path,'train.csv'))
    data = data_preprocess(data)
    data.to_csv( os.path.join( path, 'course', "train_preprocess.csv"), index=False)
    # data = pd.read_csv( os.path.join( path, 'course', "train_preprocess.csv"))

    user_id = {y:x for x, y in data['user_id'].items()}
    
    val = pd.read_csv( os.path.join( path,'val_seen.csv'))
    val = test_uid(data, val)
    val.to_csv( os.path.join( path, 'course', "val_uid.csv"),index=False)     

    val = pd.read_csv( os.path.join( path, 'test_seen.csv'))
    val = test_uid(data, val)
    val.to_csv( os.path.join( path, 'course', "test_uid.csv"),index=False)       

    df = course_feature(data, "old")
    df.to_csv( os.path.join( path, 'course', "course_feature.csv"), index=False)

    data = pd.read_csv( os.path.join( path, "courses.csv"))
    df = course_feature(data, "new")
    df.to_csv( os.path.join( path, 'course', "new_course_feature.csv"), index=False)