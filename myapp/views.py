import joblib
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework import status
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor  # Import XGBoost
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import os
from django.conf import settings
import logging

# Add this near the top of your file
logging.basicConfig(level=logging.INFO)

# Path to save the trained model
MODEL_FILE_PATH = os.path.join(settings.BASE_DIR, 'trained_model.joblib')

@api_view(['POST'])
def train_model(request):
    try:
        # Remove existing model file if it exists
        if os.path.exists(MODEL_FILE_PATH):
            os.remove(MODEL_FILE_PATH)
        
        # Get the uploaded CSV file
        csv_data = request.FILES['csv_file']
        
        # Read the CSV data into a DataFrame and filter columns
        user_data = pd.read_csv(csv_data)
        user_data = user_data[['Distance', 'Avg HR', 'Avg Pace', 'Date']]

        def string_to_year(string):
            if string:
                return int(string.split('-')[0])
            else:
                return 'Year missing'

        # Convert the 'Date' column to a column of years
        user_data['Year'] = user_data['Date'].apply(lambda x: string_to_year(x))

        # Print out unique years found in the dataset
        unique_years = user_data['Year'].unique().tolist()
        print("Years found in the dataset:", unique_years)

        # Preprocess the data to handle '--' and convert columns
        user_data['Avg HR'] = pd.to_numeric(user_data['Avg HR'].replace('--', pd.NA), errors='coerce')
        user_data['Distance'] = user_data['Distance'].str.replace(',', '.').astype(float)

        # Convert 'Avg Pace' to 'Speed (km/h)'
        def pace_to_speed(pace):
            if isinstance(pace, str):
                try:
                    minutes, seconds = map(int, pace.split(':'))
                    total_seconds = minutes * 60 + seconds
                    return 3600 / total_seconds if total_seconds > 0 else None
                except ValueError:
                    return None
            return None

        user_data['Speed (km/h)'] = user_data['Avg Pace'].apply(pace_to_speed)

        # Drop rows with any NaN values in key columns
        user_data.dropna(subset=['Avg HR', 'Distance', 'Speed (km/h)'], inplace=True)

        # Prepare features and target
        X = user_data[['Avg HR', 'Distance', 'Year']]
        y = user_data['Speed (km/h)']

        # Train a Polynomial Regression model with degree 2 on 100% of the data
        # poly_reg_model = Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("poly_features", PolynomialFeatures(degree=1)),
        #     ("lin_reg", LinearRegression())
        # ])

                # Train an XGBoost model
        xgb_model = XGBRegressor(
            n_estimators=100,  # Number of trees
            max_depth=6,       # Maximum depth of each tree
            learning_rate=0.1, # Step size shrinkage
            objective='reg:squarederror'  # Regression objective
        )
        
        # Fit the model on all available data
        # poly_reg_model.fit(X, y)
        xgb_model.fit(X, y)

        # Save the trained model
        # joblib.dump(poly_reg_model, MODEL_FILE_PATH)
        joblib.dump(xgb_model, MODEL_FILE_PATH)

        # return JsonResponse({"message": "Model trained successfully."}, status=status.HTTP_200_OK)
        return JsonResponse({"years_trained": unique_years}, status=status.HTTP_200_OK)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def predict_distances(request):
    # Predefined distances and the HR zones to use for each category
    CATEGORIZED_DISTANCES = {
        'Sprint Distances': {
            "100 m": 0.1,
            "200 m": 0.2,
            "400 m": 0.4,
            "500 m": 0.5,
            "800 m": 0.8,
            "1000 m": 1,
            "1500 m": 1.5,
            "1 mile": 1.6
        },
        'Middle Distances': {
            "2 km": 2,
            "3 km": 3,
            "2 miles": 3.2,
            "4 km": 4,
            "3 miles": 4.8,
            "5 km": 5
        },
        'Long Distances': {
            "5 miles": 8,
            "10 km": 10,
            "12 km": 12,
            "15 km": 15,
            "10 miles": 16,
            "Half marathon": 21.1,
            "Marathon": 42.2
        },
        'Ultra Distances': {
            "50 km": 50,
            "50 miles": 80,
            "100 km": 100,
            "100 miles": 160
        }
    }

    # HR zone percentages relative to HR Max for each category
    # HR_ZONE_PERCENTAGES = {
    #     'Sprint Distances': 0.95,  # Zone 5
    #     'Middle Distances': 0.85,  # Zone 4
    #     'Long Distances': 0.75,    # Zone 3
    #     'Ultra Distances': 0.65    # Zone 2
    # }

    def speed_to_time(speed, distance):
        if speed > 0:
            time_sec = (distance / speed) * 3600  # Convert time from hours to seconds
            if time_sec < 0 or time_sec > (24 * 3600):  # Add a sanity check for unrealistic values
                return "Unrealistic Time"
            hours = int(time_sec // 3600)
            minutes = int((time_sec % 3600) // 60)
            seconds = int(time_sec % 60)
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return "Invalid Speed"

    def speed_to_pace(speed):
        if speed > 0:
            pace_sec = 3600 / speed  # Convert speed (km/h) to pace (sec/km)
            minutes = int(pace_sec // 60)
            seconds = int(pace_sec % 60)
            return f"{minutes}:{seconds:02d}"
        return None

    try:
        if not os.path.exists(MODEL_FILE_PATH):
            return JsonResponse({"error": "No trained model found. Please upload data and train the model first."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Load the model
        model = joblib.load(MODEL_FILE_PATH)

        # Get HR Max from request
        hr_zones = request.data['zones']  # Dictionary containing HR for each category
        year = int(request.data['year'])

        predictions = {'Sprint Distances': [], 'Middle Distances': [], 'Long Distances': [], 'Ultra Distances': []}

        for category, distances in CATEGORIZED_DISTANCES.items():
            if category not in hr_zones:
                return JsonResponse({"error": f"HR zone for {category} not provided."}, status=status.HTTP_400_BAD_REQUEST)
            
            hr_for_category = hr_zones[category]  # Use the provided HR for this category

            for distance_title, distance in distances.items():
                input_data = pd.DataFrame({'Avg HR': [hr_for_category], 'Distance': [distance], 'Year': [year]})
                predicted_speed = model.predict(input_data)[0]
                
                # Check for unrealistic speeds
                if predicted_speed < 1 or predicted_speed > 50:
                    logging.warning(f"Unrealistic predicted speed for {distance_title}: {predicted_speed:.2f} km/h")

                predicted_pace = speed_to_pace(predicted_speed)
                predicted_time = speed_to_time(predicted_speed, distance)

                predictions[category].append({
                    'distance_title': distance_title,
                    'distance': distance,
                    'predicted_pace': predicted_pace,
                    'finish_time': predicted_time,
                    'applied_hr': round(hr_for_category),
                    'context_year': year
                })

        return JsonResponse({'predictions': predictions}, status=status.HTTP_200_OK)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
