import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Load and preprocess data
booking = pd.read_csv("data/BookingDetails.csv", low_memory=False)
selected_cols = ['age', 'sex', 'region', 'travel_type_long', 'stars', 'board1']
booking_cluster = booking[selected_cols].dropna()

# Convert 'stars' to float (handle commas)
booking_cluster.loc[:, 'stars'] = booking_cluster['stars'].str.replace(',', '.').astype(float)

# Encode 'sex' as numeric
booking_cluster.loc[:, 'sex'] = booking_cluster['sex'].map({'M': 0, 'F': 1, 'C': 2})

# One-hot encode categorical features
booking_encoded = pd.get_dummies(booking_cluster, columns=['region', 'travel_type_long', 'board1'])

# Step 2: Scale features for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(booking_encoded)

# Step 3: Run KMeans clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Step 4: Add cluster labels to original data
booking_cluster['Cluster'] = cluster_labels

# Step 5: Save to CSV for further use
booking_cluster.to_csv("clustered_customers.csv", index=False)
print("✅ Clustered data saved to clustered_customers.csv")

# Step 6: Show basic cluster sizes
print("\n📊 Cluster sizes:")
print(booking_cluster['Cluster'].value_counts().sort_index())

# Step 7: Numeric profile per cluster
print("\n📈 Averages per cluster:")
print(booking_cluster.groupby('Cluster')[['age', 'stars', 'sex']].mean())

# Step 8: Most common values per cluster (region, travel type, board)
print("\n🏖️ Top regions per cluster:")
print(booking_cluster.groupby('Cluster')['region'].agg(lambda x: x.value_counts().index[0]))

print("\n🧳 Most common travel type per cluster:")
print(booking_cluster.groupby('Cluster')['travel_type_long'].agg(lambda x: x.value_counts().index[0]))

print("\n🍽️ Most common board type per cluster:")
print(booking_cluster.groupby('Cluster')['board1'].agg(lambda x: x.value_counts().index[0]))

