"""Run this script after running the notebook to save the trained model."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

RANDOM_STATE = 42

df = pd.read_csv('data/ames_housing.csv')

X = df[['GrLivArea', 'OverallQual', 'Neighborhood']]
y = df['SalePrice']

X = pd.get_dummies(X, columns=['Neighborhood'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')
print('Saved model.pkl and model_columns.pkl')
