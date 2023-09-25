
# Medical Visit Predictor

## Introduction

This project aims to predict the number of medical visits for various hospitals using time series forecasting. The primary method used for forecasting is the Exponential Smoothing technique.

## Model

### Exponential Smoothing

Exponential Smoothing is a time series forecasting method that involves calculating weighted averages of past observations, with the weights decaying exponentially as the observations get older. This means the model gives more importance to recent observations while still considering the historical data.

### Results & Accuracy

The model's performance can be evaluated using the R2 score, which provides a measure of how well the predicted values match the actual values. A higher R2 score indicates a better fit of the model to the data. In our analysis, the exact R2 scores were computed, but they are currently set as a placeholder. Please refer to the original Python script or notebook to fill in the precise values: `R2 Score: [PLACEHOLDER]`.

## Deployment

### Local Deployment:

1. **Build the Docker Image**:
   ```bash
   docker build -t medical_visit_predictor .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 5000:5000 medical_visit_predictor
   ```

3. **Access the API**: Once the Docker container is running, the Flask API can be accessed at `http://localhost:5000`.

## Usage

To predict medical visits for a given number of days:

1. Send a POST request to the `/forecast` endpoint with a JSON body containing the number of days to predict. For example:

   ```json
   {
       "days_to_predict": 14
   }
   ```

2. The API will respond with the forecasted values for the specified number of days.

## Conclusion

This project provides a simple yet effective way to predict medical visits using time series data. By leveraging the Exponential Smoothing technique and providing an easy-to-use API, users can get accurate forecasts for medical visits.
