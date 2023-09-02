import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv('telecom.csv')

# List of columns to drop
columns_to_drop = [
    'customerID',
    'gender',
    'SeniorCitizen',
    'Dependents',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod',
]

# Drop the columns
df = df.drop(columns=columns_to_drop)

# Convert binary 'Yes'/'No' columns to 0/1
binary_columns_to_convert = ['Partner', 'PhoneService', 'PaperlessBilling', 'Churn']
df[binary_columns_to_convert] = df[binary_columns_to_convert].apply(lambda x: x.map({'No': 0, 'Yes': 1}))

# Ensure 'TotalCharges' is numeric and replace missing values with 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

# Split data
X = df.drop(['Churn'], axis=1)
y1 = df['TotalCharges']
y2 = df['Churn']

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.3, random_state=42)

# Train TotalCharges model
model1 = LinearRegression()
model1.fit(X_train, y1_train)

# Evaluate TotalCharges model
y1_pred = model1.predict(X_test)
r2 = r2_score(y1_test, y1_pred)
print("R-squared for TotalCharges:", r2)

# Train Churn model
model2 = LinearRegression()
model2.fit(X_train, y2_train)

# Evaluate Churn model
y2_pred = model2.predict(X_test)
mse = mean_squared_error(y2_test, y2_pred)
print("MSE for Churn:", mse)

corr = df.corr()
print("Correlations for Remaining Columns:")
print(corr['TotalCharges'].sort_values(ascending=False))
print(corr['Churn'].sort_values(ascending=False))