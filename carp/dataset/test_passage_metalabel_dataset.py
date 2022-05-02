import pandas as pd

data = pd.read_csv("passage_metalabel_dataset.csv", index_col=0)
print(data.head())

class_distributions = data[[str(i) for i in range(92)]]
print(class_distributions.shape)

passages = list(data["passages"])
reviews = class_distributions.values.tolist()
print(len(passages), len(reviews))

first_class = list(data["0"])
print(len(first_class))
