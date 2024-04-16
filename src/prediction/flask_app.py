from cloudpickle import load

from flask import Flask, request, jsonify
import pandas as pd
from src.utils.constant import model_local_path

app = Flask(__name__)


@app.route('/health', methods=["GET"])
def health_check():
    # Health check endpoint
    return "Ok"


@app.route('/predict', methods=["POST"])
def predict_pulsar_json():
    """
    Endpoint to predict if an input in JSON format contains house price data.
    ---
    parameters:
      - name: input_json
        in: body
        type: application/json
        required: true
    responses:
      200:
        description: The predicted value
    """
    try:
        # Extracting input parameters from the JSON request
        input_data = request.get_json()

        # Convert input data to DataFrame
        df_test = pd.DataFrame.from_dict(input_data, orient='columns')

        # Making predictions using the loaded classifier
        prediction = model.predict(df_test)

        # # Convert the NumPy array to a list and process it to remove unwanted elements
        # prediction_list = prediction.tolist()  # Convert to list
        # cleaned_predictions = [float(pred[0]) for pred in prediction_list]  # Remove array notation and dtype
        #
        # # Define the threshold for classification
        # threshold = 0.5
        #
        # # Apply the threshold to the predictions and create a list of dictionaries with serial numbers
        # predicted_chances = [{'For input instance': i + 1, 'Person belongs to': 'Medium or High Risk Class' if pred >=
        #                                                                                                        threshold else 'Low Risk Class'}
        #                      for i, pred in enumerate(cleaned_predictions)]

        # Return the predicted value in JSON format
        return jsonify({"predicted_chances": str(list(prediction))})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Load the model_for_flask_app
    model = load(open(model_local_path, 'rb'))
    # Run the Flask app on all available network interfaces
    app.run(host='0.0.0.0', port=8000)