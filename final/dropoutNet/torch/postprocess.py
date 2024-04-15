import pandas as pd



def remove_seen(file, val):

    train = pd.read_csv(file)
    user_id = {y: x for x, y in train['user_id'].items()}
    for i in range(len(val)):
        text = train['course_id'][user_id[val['user_id'][i]]]
        _pred = []
        p = val['course_id'][i].split(" ")    
        for j in range(len(p)):
            if p[j] not in text:
                _pred.append(p[j])
        val['course_id'][i] = " ".join(str(p) for p in _pred)
    return val

def zero(val, course):
    ID = [682, 681, 690, 48, 79, 80, 42, 673, 5, 25, 24, 23, 14, 12, 13, 20]
    courses = course['course_id'][ID].tolist()

    for i in range(len(val)):
        _pred = courses.copy()
        p = val['course_id'][i].split(" ")    



        for j in range(len(p)):
            if p[j] not in _pred:
                _pred.append(p[j])
        
        val['course_id'][i] = " ".join(str(p) for p in _pred)
    return val