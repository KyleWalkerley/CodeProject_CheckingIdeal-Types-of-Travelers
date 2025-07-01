import pandas as pd

# Reload or use cleaned dataset
booking = pd.read_csv("data/BookingDetails.csv", low_memory=False)
selected_cols = ['age', 'sex', 'region', 'travel_type_long', 'stars', 'board1']
booking_cluster = booking[selected_cols].dropna()

# Step 1: Clean and convert 'stars' to float
# Replace commas with dots and convert to float
booking_cluster.loc[:, 'stars'] = booking_cluster['stars'].str.replace(',', '.').astype(float)

# Step 2: Encode 'sex' using simple mapping
sex_map = {'M': 0, 'F': 1, 'C': 2}
booking_cluster['sex'] = booking_cluster['sex'].map(sex_map)

# Step 3: One-hot encode 'region', 'travel_type_long', and 'board1'
booking_encoded = pd.get_dummies(booking_cluster, columns=['region', 'travel_type_long', 'board1'])

# Step 4: Final check
print("✅ Ready for clustering — encoded shape:", booking_encoded.shape)
print("🔢 Sample columns:", booking_encoded.columns[:10])
