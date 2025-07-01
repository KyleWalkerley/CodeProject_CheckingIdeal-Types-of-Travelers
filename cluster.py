import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the preprocessed dataset
booking = pd.read_csv("data/BookingDetails.csv", low_memory=False)
selected_cols = ['age', 'sex', 'region', 'travel_type_long', 'stars', 'board1']
booking_cluster = booking[selected_cols].dropna()

# Convert and encode (same as main.py)
booking_cluster.loc[:, 'stars'] = booking_cluster['stars'].str.replace(',', '.').astype(float)
booking_cluster.loc[:, 'sex'] = booking_cluster['sex'].map({'M': 0, 'F': 1, 'C': 2})
booking_encoded = pd.get_dummies(booking_cluster, columns=['region', 'travel_type_long', 'board1'])

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(booking_encoded)

# Elbow Method to find optimal cluster count
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()
