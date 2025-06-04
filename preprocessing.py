import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("heart_cleveland_upload.csv")
print(f"Dataset shape: {df.shape}")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Separate features and target
X = df.drop('condition', axis=1)
y = df['condition']

# Define preprocessing steps
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
print(f"\nProcessed features shape: {X_processed.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]} samples")
print(f"Test size: {X_test.shape[0]} samples")

# Save preprocessor and data
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump((X_train, X_test, y_train, y_test), 'train_test_data.pkl')

print("\nPreprocessing complete! Files saved.")

# In preprocessing.py
print("Original columns:", list(df.drop('condition', axis=1).columns))