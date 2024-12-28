import joblib
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework import status
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
import os
from django.conf import settings

# Path to save the trained model
MODEL_FILE_PATH = os.path.join(settings.BASE_DIR, 'trained_model.joblib')

@api_view(['POST'])
def train_model(request):
    try:
        if os.path.exists(MODEL_FILE_PATH):
            os.remove(MODEL_FILE_PATH)
        # Get the uploaded CSV file
        csv_data = request.FILES['csv_file']
        
        # Read the CSV data into a DataFrame and filter columns
        user_data = pd.read_csv(csv_data)
        user_data = user_data[['Distance', 'Avg HR', 'Avg Pace']]

        # Preprocess the data
        user_data['Avg HR'] = pd.to_numeric(user_data['Avg HR'].replace('--', pd.NA), errors='coerce')
        user_data['Distance'] = user_data['Distance'].str.replace(',', '.').astype(float)

        # Function to convert "mm:ss" pace format to seconds
        def pace_to_seconds(pace):
            if isinstance(pace, str):
                try:
                    minutes, seconds = map(int, pace.split(':'))
                    return minutes * 60 + seconds
                except ValueError:
                    return None
            return None

        user_data['Avg Pace (sec)'] = user_data['Avg Pace'].apply(pace_to_seconds)
        user_data.dropna(subset=['Avg HR', 'Distance', 'Avg Pace (sec)'], inplace=True)

        # Prepare features and target (pace in seconds per km)
        X = user_data[['Avg HR', 'Distance']]
        y = user_data['Avg Pace (sec)']

        # Train the Polynomial Regression model on pace instead of speed
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("poly_features", PolynomialFeatures(degree=2)),
            ("lin_reg", LinearRegression())
        ])
        
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE_PATH)

        return JsonResponse({"message": "Model trained successfully."}, status=status.HTTP_200_OK)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def predict_distances(request):
    # Predefined distances to predict
    PREDEFINED_DISTANCES = [5, 10, 15, 21.1, 42.2, 50]

    # Convert predicted pace in seconds per km back to "mm:ss" format
    def seconds_to_pace(seconds):
        if seconds > 0:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}:{secs:02d}"
        return None

    try:
        # Check if the model file exists
        if not os.path.exists(MODEL_FILE_PATH):
            return JsonResponse({"error": "No trained model found. Please upload data and train the model first."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Load the model
        model = joblib.load(MODEL_FILE_PATH)

        # Get heart rate from request
        heart_rate = float(request.data['heart_rate'])
        
        predictions = []
        for distance in PREDEFINED_DISTANCES:
            input_data = pd.DataFrame({'Avg HR': [heart_rate], 'Distance': [distance]})
            predicted_pace_seconds = model.predict(input_data)[0]
            predicted_pace = seconds_to_pace(predicted_pace_seconds)
            predictions.append({'distance': distance, 'predicted_pace': predicted_pace})

        return JsonResponse({'predictions': predictions}, status=status.HTTP_200_OK)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
