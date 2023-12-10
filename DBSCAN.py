import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

url = "C:\\Users\\solo\\Desktop\\iris.csv"
df = pd.read_csv(url)
print(df.head())
data = df[["sepal.length", "sepal.width"]]

model = DBSCAN(eps=0.4, min_samples=10).fit(data)
colors = model.labels_
plt.scatter(data["sepal.length"], data["sepal.width"], c=colors)
outliers = data[model.labels_ == -1]

print(outliers)