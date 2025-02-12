import pandas as pd
import joblib

# Load the trained model from the saved file
model = joblib.load('/home/vigneshkumar/Desktop/fun/myenv/loan_prediction/loan_prediction_model.pkl')
print("Model loaded successfully.")

# New data (input the soil parameter values)
new_data = pd.DataFrame({
    'pH': [14],  # Example pH value
    'Moisture (%)': [35.0],  # Example moisture value
    'Temperature (°C)': [18.5],  # Example temperature value
    'Conductivity (dS/m)': [1.2],  # Example conductivity value
    'Salinity (ppm)': [450],  # Example salinity value
    'Humidity (%)': [70],  # Example humidity value
    'Pressure (hPa)': [1013],  # Example pressure value
    'Temperature (Air) (°C)': [22.0]  # Example air temperature value
})

# Predict the loan amount for the new data
predicted_loan = model.predict(new_data)

# Print the predicted loan amount
print(f'Predicted Loan Amount: {predicted_loan[0]}')
