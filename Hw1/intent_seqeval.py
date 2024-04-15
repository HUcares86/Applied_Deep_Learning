from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import csv
import pandas as pd
import os

df = pd.read_csv('/home/shelley/Desktop/hucares/ADL21-HW1-main/pred.intent.csv')
print(df)
df=df.drop(columns=['id'])
print(df)
df.to_csv(os.path.join('/home/shelley/Desktop/hucares/ADL21-HW1-main/','pred.intent_seqeval.csv'),index=False) #save to file
with open('/home/shelley/Desktop/hucares/ADL21-HW1-main/pred.intent_seqeval.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)


print(data)


import json
training_seqeval = []
f = open('/home/shelley/Desktop/hucares/ADL21-HW1-main/data/intent/train.json')
a = json.load(f)
print(len(a))
for i in range(len(a)):
    training_seqeval.append([str(a[i]['intent'])])
    #print([str(a[i]['intent'])])

# print(training_seqeval)

# intent_f1_score = f1_score(training_seqeval, data)
# intent_classification_report = classification_report(training_seqeval, data)

