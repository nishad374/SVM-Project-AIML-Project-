from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib
import time

# Load preprocessed data
X_train, _, y_train, _ = joblib.load('train_test_data.pkl')
print(f"Training data shape: {X_train.shape}")

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

print("\nStarting GridSearchCV...")
start_time = time.time()

svm = SVC(probability=True, random_state=42)
grid_search = GridSearchCV(
    svm, 
    param_grid, 
    cv=5, 
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nTraining completed in {time.time()-start_time:.2f} seconds")

# Save best model
best_svm = grid_search.best_estimator_
joblib.dump(best_svm, 'svm_model.pkl')

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best ROC-AUC score: {grid_search.best_score_:.4f}")
print("Model saved as 'svm_model.pkl'")