import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
n_samples = 1000
n_features = 10
# Define activities
activities = ['walking', 'running', 'sitting']
n_classess = len(activities)

# Generate synthetic data
data = []
for activity in activities:
    for _ in range(n_samples // n_classess):
        if activity == 'walking':
            features = np.random.normal(loc=0.5, scale=0.1, size=n_features)
        elif activity == 'running':
            features = np.random.normal(loc=2.0, scale=0.2, size=n_features)  # Added missing parameters
        else:
            features = np.random.normal(loc=1.0, scale=0.1, size=n_features)
        data.append(np.append(features, activity))
# Create DataFrame
columns = [f'feature_{i + 1}' for i in range(n_features)] + ['activity']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('activity_data.csv', index=False)

print("Synthetic data for activity classification generated and saved to 'activity_data.csv'!")
