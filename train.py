import pandas as pd
import xgboost as xgb
import joblib

# Here are your scikit-learn imports!
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

print("1. Creating dataset...")
df = pd.DataFrame({
    'startingAirport': ['JFK', 'LAX', 'ORD', 'DFW', 'ATL'] * 2000,
    'destinationAirport': ['LAX', 'JFK', 'DFW', 'ORD', 'MIA'] * 2000,
    'isBasicEconomy': [1, 0, 1, 0, 1] * 2000,
    'isRefundable': [0, 1, 0, 1, 0] * 2000,
    'isNonStop': [1, 1, 0, 0, 1] * 2000,
    'totalFare': [350.0, 420.0, 200.0, 190.0, 150.0] * 2000
})

print("2. Processing data...")
airport_map = {'JFK': 0, 'LAX': 1, 'ORD': 2, 'DFW': 3, 'ATL': 4, 'MIA': 5}
df['startingAirport'] = df['startingAirport'].map(airport_map).fillna(-1)
df['destinationAirport'] = df['destinationAirport'].map(airport_map).fillna(-1)

X = df[['startingAirport', 'destinationAirport', 'isBasicEconomy', 'isRefundable', 'isNonStop']]
y = df['totalFare']

print("3. Splitting data with scikit-learn...")
# Using scikit-learn to split 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Training XGBoost Model...")
# The XGBRegressor uses scikit-learn's API under the hood
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

print("5. Evaluating model with scikit-learn...")
# Using scikit-learn to see how accurate our model is
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"Model Error Margin: ${error:.2f}")

print("6. Saving model...")
joblib.dump(model, 'xgboost_model.pkl')
joblib.dump(airport_map, 'airport_map.pkl')
print("Done! You can now start your API.")