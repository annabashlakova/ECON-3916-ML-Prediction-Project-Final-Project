import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Ames Housing Sale Price Predictor')
st.markdown('Predict a home\'s expected sale price based on its attributes, condition, and location.')

# Load model and column names
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Sidebar controls
st.sidebar.header('Home Features')

gr_liv_area = st.sidebar.slider('Above-Ground Living Area (sq ft)', 334, 5642, 1500)
overall_qual = st.sidebar.slider('Overall Quality (1-10)', 1, 10, 5)

neighborhoods = [
    'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
    'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
    'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown',
    'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'
]
neighborhood = st.sidebar.selectbox('Neighborhood', neighborhoods)

# Build input dataframe
input_data = pd.DataFrame({
    'GrLivArea': [gr_liv_area],
    'OverallQual': [overall_qual],
    'Neighborhood': [neighborhood]
})

# One-hot encode to match training
input_encoded = pd.get_dummies(input_data, columns=['Neighborhood'], drop_first=True)

# Align columns with training data
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_columns]

# Predict
prediction = model.predict(input_encoded)[0]

# Get predictions from all trees for uncertainty
tree_preds = np.array([tree.predict(input_encoded)[0] for tree in model.estimators_])
lower = np.percentile(tree_preds, 10)
upper = np.percentile(tree_preds, 90)

# Display
st.metric('Predicted Sale Price', f'${prediction:,.0f}')
st.markdown(f'**80% Prediction Interval:** ${lower:,.0f} — ${upper:,.0f}')

# Visualization
st.subheader('Prediction Distribution Across Trees')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(tree_preds, bins=30, color='steelblue', edgecolor='white')
ax.axvline(prediction, color='red', linestyle='--', label=f'Prediction: ${prediction:,.0f}')
ax.axvline(lower, color='orange', linestyle=':', label=f'10th pctile: ${lower:,.0f}')
ax.axvline(upper, color='orange', linestyle=':', label=f'90th pctile: ${upper:,.0f}')
ax.set_xlabel('Predicted Sale Price ($)')
ax.set_ylabel('Number of Trees')
ax.legend()
plt.tight_layout()
st.pyplot(fig)
