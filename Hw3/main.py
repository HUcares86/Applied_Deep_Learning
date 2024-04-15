from tw_rouge import get_rouge


import csv

with open("/home/shelley/Desktop/hucares/Hw3/predsNo_trianer_Top_k_5.csv", newline='') as f:
    reader = csv.reader(f)
    preds = list(reader)[1]

# print(preds)

with open("/home/shelley/Desktop/hucares/Hw3/predsNo_trianer_Top_k_5.csv", newline='') as f:
    reader = csv.reader(f)
    labels = list(reader)[1]

result = get_rouge(preds, labels)

print("get_rouge end")
print(result)
print("-----")
result = {f"{k1}_{k2}": v2  for k1, v1 in result.items() for k2, v2 in v1.items()}











