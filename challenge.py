import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = {
    "Age": [21, 45, 23, 42, 19, 53, 24, 34, 29, 37, 30, 41, 31, 59, 61],
    "Gender": ["F", "M", "M", "M", "F", "F", "M", "F", "F", "M", "F", "M", "F", "M", "F"],
    "Monthly Income": [210000, 430000, 243000, 542000, 670000, 321000, 423000, 217000, 452000, 413000, 736000, 652000, 298000, 900000, 198000],
    "Number of Purchases": [2, 1, 6, 7, 6, 2, 5, 5, 9, 1, 4, 6, 1, 5, 7],
    "Total Amount Spent": [20000, 30900, 32000, 37000, 65000, 21000, 42000, 65400, 54100, 50000, 70000, 54100, 53000, 29000, 41000],
}

df = pd.DataFrame(data)
print(df)

df["Gender_Encoded"] = df["Gender"].map({"F": 0, "M": 1})

print(df)

features = df[["Age", "Gender_Encoded", "Monthly Income", "Number of Purchases", "Total Amount Spent"]]
print(features)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print(features_scaled)

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
df["Cluster"] = kmeans.fit_predict(features_scaled)

print(df)

print(df.groupby("Cluster")[["Age", "Gender_Encoded", "Monthly Income", "Number of Purchases", "Total Amount Spent"]].mean())

features_no_gender = df[["Age", "Monthly Income", "Number of Purchases", "Total Amount Spent"]]

scaler2 = StandardScaler()
features_no_gender_scaled = scaler2.fit_transform(features_no_gender)

kmeans2 = KMeans(n_clusters=3, random_state=0, n_init=10)
df["Segment_No_Gender"] = kmeans2.fit_predict(features_no_gender_scaled)

print(df.groupby("Segment_No_Gender")[["Age", "Monthly Income", "Number of Purchases", "Total Amount Spent"]].mean())
print("\n Gender breakdown per segment:")
print(df.groupby("Segment_No_Gender")["Gender_Encoded"].value_counts())