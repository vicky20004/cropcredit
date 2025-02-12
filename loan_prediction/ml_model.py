
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load the dataset (replace 'your_file.csv' with your actual file path)
df = pd.read_csv('/home/vigneshkumar/Desktop/fun/myenv/loan_prediction/soil_data.csv')

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Feature columns (predictors)
X = df[['pH', 'Moisture (%)', 'Temperature (°C)', 'Conductivity (dS/m)', 
        'Salinity (ppm)', 'Humidity (%)', 'Pressure (hPa)', 'Temperature (Air) (°C)']]

# Target variable (the loan amount)
y = df['Loan']

# Split the dataset into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean absolute error (MAE) to evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the trained model to a file (using joblib)
joblib.dump(model, '/home/vigneshkumar/Desktop/fun/myenv/loan_prediction/loan_prediction_model.pkl')
print("Model saved as 'loan_prediction_model.pkl'")
