import pandas as pd
import joblib
import xgboost as xgb
from flask import Flask, request, jsonify, render_template

# --- Libraries needed for model loading (must be imported globally) ---
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline as ImbPipeline 
# NOTE: The imports above MUST match the imports used in fix_model.py

app = Flask(__name__)

# Global variables to hold the loaded model and preprocessor
model = None
preprocessor = None
RESOURCES_LOADED = False

def load_resources():
    """
    Loads the XGBoost model and the ColumnTransformer preprocessor from joblib files.
    This function includes the necessary imports to handle joblib deserialization.
    """
    global model
    global preprocessor
    global RESOURCES_LOADED

    # NOTE: These paths must match the file names created by fix_model.py
    PREPROCESSOR_PATH = 'preprocessor (1).joblib'
    MODEL_PATH = 'smote_xgboost_model (1).joblib'

    try:
        # Load the preprocessor (ColumnTransformer)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        # Load the final trained model (ImbPipeline containing XGBClassifier)
        model = joblib.load(MODEL_PATH)
        
        RESOURCES_LOADED = True
        print("\n--- Models and Data Loaded Successfully ---\n")

    except Exception as e:
        RESOURCES_LOADED = False
        print("\nFATAL ERROR: Failed to Load Resources! Details:", e)
        print("\n🚨 VERSION CONFLICT: Please re-run your training script (fix_model.py) to save compatible joblib files.")
        
# Load resources immediately when the Flask app starts
with app.app_context():
    load_resources()

@app.route('/')
def home():
    """Renders the HTML form for user input."""
    if not RESOURCES_LOADED:
        return render_template('error.html', message="Server Error: Model failed to load. Please check logs.")
    
    # This mock HTML template is embedded directly into the response
    # FIX: We use standard JavaScript string concatenation to avoid Python syntax issues with backticks.
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fraud Detection Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #f7fafc;
            }}
            .card {{
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            }}
        </style>
    </head>
    <body class="min-h-screen flex items-center justify-center p-4">
        <div class="card bg-white p-8 md:p-10 rounded-xl w-full max-w-lg">
            <h1 class="text-3xl font-bold text-gray-800 mb-6 text-center">Fraud Risk Predictor</h1>
            
            <!-- Result Display Box -->
            <div id="result-box" class="hidden p-4 mb-6 rounded-lg font-semibold text-center transition-all duration-300"></div>

            <form id="prediction-form" class="space-y-4">

                <!-- Row 1: Amount -->
                <div>
                    <label for="amount" class="block text-sm font-medium text-gray-700 mb-1">Transaction Amount (INR)</label>
                    <input type="number" step="0.01" id="amount" name="amount" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 transition duration-150">
                </div>

                <!-- Row 2: State / Card Type -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label for="state" class="block text-sm font-medium text-gray-700 mb-1">State</label>
                        <select id="state" name="state" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 transition duration-150">
                            <option value="Telangana">Telangana (Sample)</option>
                            <option value="Maharashtra">Maharashtra</option>
                            <option option value="Uttar Pradesh">Uttar Pradesh</option>
                            <option value="Delhi">Delhi</option>
                            <option value="Tamil Nadu">Tamil Nadu</option>
                            <option value="West Bengal">West Bengal</option>
                            <option value="Karnataka">Karnataka</option>
                            <option value="Gujarat">Gujarat</option>
                            <option value="Kerala">Kerala</option>
                            <option value="Assam">Assam</option>
                            <!-- Add all unique states from your dataset here -->
                        </select>
                    </div>
                    <div>
                        <label for="card_type" class="block text-sm font-medium text-gray-700 mb-1">Card Type</label>
                        <select id="card_type" name="card_type" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 transition duration-150">
                            <option value="Rupay">Rupay (Sample)</option>
                            <option value="Visa">Visa</option>
                            <option value="Mastercard">Mastercard</option>
                            <option value="Amex">Amex</option>
                        </select>
                    </div>
                </div>

                <!-- Row 3: Bank / Category -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label for="bank" class="block text-sm font-medium text-gray-700 mb-1">Bank</label>
                        <select id="bank" name="bank" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 transition duration-150">
                            <option value="ICICI Bank">ICICI Bank (Sample)</option>
                            <option value="SBI">SBI</option>
                            <option value="HDFC Bank">HDFC Bank</option>
                            <option value="Federal Bank">Federal Bank</option>
                            <option value="Axis Bank">Axis Bank</option>
                            <option value="Andhra Bank">Andhra Bank</option>
                            <!-- Add all unique banks from your dataset here -->
                        </select>
                    </div>
                    <div>
                        <label for="category" class="block text-sm font-medium text-gray-700 mb-1">Category</label>
                        <select id="category" name="category" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 transition duration-150">
                            <option value="Transportation">Transportation (Sample)</option>
                            <option value="Groceries">Groceries</option>
                            <option value="E-commerce">E-commerce</option>
                            <option value="Electronics">Electronics</option>
                            <option value="Food Delivery">Food Delivery</option>
                            <!-- Add all unique categories from your dataset here -->
                        </select>
                    </div>
                </div>

                <!-- Row 4: Merchant Location -->
                <div>
                    <label for="location" class="block text-sm font-medium text-gray-700 mb-1">Merchant Location</label>
                    <select id="location" name="location" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 transition duration-150">
                        <option value="Bangalore">Bangalore (Sample)</option>
                        <option value="Mumbai">Mumbai</option>
                        <option value="Ahmedabad">Ahmedabad</option>
                        <option value="Kolkata">Kolkata</option>
                        <option value="Hyderabad">Hyderabad</option>
                        <option value="Lucknow">Lucknow</option>
                        <option value="Chennai">Chennai</option>
                        <option value="Delhi">Delhi</option>
                        <!-- Add all unique locations from your dataset here -->
                    </select>
                </div>

                <!-- Submit Button -->
                <button type="submit" id="submit-button" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50">
                    Predict Fraud Risk
                </button>
            </form>
        </div>

        <script>
            document.getElementById('prediction-form').addEventListener('submit', async function(e) {{
                e.preventDefault();

                const form = e.target;
                const formData = new FormData(form);
                const data = {{}};
                formData.forEach((value, key) => {{ data[key] = value; }});
                
                const resultBox = document.getElementById('result-box');
                const submitButton = document.getElementById('submit-button');
                
                // Show loading state
                submitButton.textContent = 'Predicting...';
                submitButton.disabled = true;
                resultBox.classList.add('hidden');

                try {{
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(data)
                    }});

                    const result = await response.json();
                    
                    if (response.ok) {{
                        // Success handling
                        const prediction = result.prediction;
                        const probability = result.probability;
                        
                        let message = '';
                        let bgColor = '';
                        let textColor = 'text-white';

                        if (prediction === 1) {{
                            message = `FRAUD DETECTED! Risk Score: ${{(probability * 100).toFixed(2)}}%`;
                            bgColor = 'bg-red-600';
                        }} else {{
                            message = `Transaction is SAFE. Risk Score: ${{(probability * 100).toFixed(2)}}%`;
                            bgColor = 'bg-green-600';
                        }}

                        // FIXED: Using string concatenation instead of template literals
                        resultBox.className = 'p-4 mb-6 rounded-lg font-semibold text-center ' + bgColor + ' ' + textColor + ' block';
                        resultBox.innerHTML = message;

                    }} else {{
                        // Server-side error handling (e.g., 500 status)
                        resultBox.className = 'p-4 mb-6 rounded-lg font-semibold text-center bg-yellow-500 text-gray-800 block';
                        resultBox.innerHTML = `Error: ${{(result.error || 'Prediction failed due to server error.')}}`;
                    }}

                }} catch (error) {{
                    // Network or JSON parsing error
                    resultBox.className = 'p-4 mb-6 rounded-lg font-semibold text-center bg-red-400 text-white block';
                    resultBox.innerHTML = 'A network error occurred. Check server status.';
                }} finally {{
                    // Reset loading state
                    submitButton.textContent = 'Predict Fraud Risk';
                    submitButton.disabled = false;
                }}
            }});
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the web form."""
    global model
    global preprocessor
    global RESOURCES_LOADED

    if not RESOURCES_LOADED or model is None or preprocessor is None:
        return jsonify({'error': 'Model resources not fully loaded on server.'}), 500

    try:
        data = request.get_json(force=True)
        
        # --- FIX: Explicitly convert 'Transaction Amount (INR)' to float ---
        try:
            data['amount'] = float(data['amount'])
        except ValueError:
            return jsonify({'error': 'Invalid format for Transaction Amount. Must be a number.'}), 400
        # ------------------------------------------------------------------

        # Create a DataFrame from the incoming JSON data
        # Feature order MUST match the training data used in fix_model.py
        input_data = pd.DataFrame([{
            'Transaction Amount (INR)': data['amount'], # Use the converted float value
            'State': data['state'], 
            'Card Type': data['card_type'], 
            'Bank': data['bank'], 
            'Transaction Category': data['category'], 
            'Merchant Location': data['location']
        }])
        
        # 1. Preprocess the data using the loaded preprocessor
        X_processed = preprocessor.transform(input_data)
        
        # 2. Make Prediction
        prediction = model.predict(X_processed)[0]
        prediction_proba = model.predict_proba(X_processed)[0, 1] # Probability of being class 1 (Fraud)

        return jsonify({
            'prediction': int(prediction),
            'probability': float(prediction_proba)
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

# The standard way to run a Flask app
if __name__ == '__main__':
    # We must explicitly call app.run() to start the server.
    app.run(debug=True, host='0.0.0.0', port=5000)



