
from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime, timedelta

# Assuming the exponential smoothing function and other necessary functions
# are defined in the exponential_smoothing.py script
from exponential_smoothing import expo_smoothing

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.json
        days_to_predict = data.get('days_to_predict', 14)  # Default to 14 days if not provided
        end_date = datetime.now() + timedelta(days=days_to_predict)
        
        # Load the dataset and filter it based on the end date
        Y_df = pd.read_csv('./datasets/combined.csv')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        Y_df = Y_df[Y_df['ds'] <= end_date.strftime('%Y-%m-%d')]
        Y_df.fillna(0, inplace=True)
        
        hosp_id = Y_df['unique_id'].iloc[0]
        
        # Call the exponential smoothing function to get the forecast
        result = expo_smoothing(hosp_id, Y_df, predict_step=days_to_predict)
        
        response = {
            "message": "Forecasting successful for hospital " + str(hosp_id),
            "forecast": result  # Assuming the expo_smoothing function returns the forecasted values
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "API is working!"}), 200

if __name__ == '__main__':
    app.run(debug=True)
