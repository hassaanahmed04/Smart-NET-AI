from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from helpers.geojson_helper import df_to_geojson
from helpers.cluster_helper import determine_optimal_clusters, cluster_and_optimize
import json

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to classify traffic
def label_traffic(row):
    if row['download_speed'] > 10 and row['upload_speed'] > 10 and row['latency'] < 50:
        return 'High-Speed Traffic'
    elif row['download_speed'] < 3 or row['upload_speed'] < 3 or row['latency'] > 100:
        return 'Low-Speed Traffic'
    elif row['latency'] > 150:
        return 'High Latency'
    else:
        return 'Normal Traffic'

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")
@app.route("/real_time_traffic_monitoring_and_prediction", methods=["GET", "POST"])
def real_time_traffic_monitoring_and_prediction():
    return render_template("real_time_traffic_monitoring_and_prediction.html")


def convert_to_geojson(df, lat_col, lon_col, props):
    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row[lon_col], row[lat_col]]
            },
            "properties": {prop: row[prop] for prop in props}
        }
        features.append(feature)
    return {
        "type": "FeatureCollection",
        "features": features
    }
@app.route("/optimal-tower-location", methods=["GET", "POST"])
def optimal_tower_location():
    if request.method == "POST":
        schools_file = request.files['schools']
        schools_path = os.path.join(app.config['UPLOAD_FOLDER'], schools_file.filename)
        schools_file.save(schools_path)

        # Load the schools file
        schools = pd.read_csv(schools_path)
        cells_file = request.files['towers']

        cells_path = os.path.join(app.config['UPLOAD_FOLDER'], cells_file.filename)
        cells_file.save(cells_path)

        # Load the cells file
        cells = pd.read_csv(cells_path)
        schools_geojson = convert_to_geojson(
            schools, "latitude", "longitude", ["school_name_x"]
        )
        cell_towers_geojson = convert_to_geojson(
            cells, "lat", "lon", ["cell"]
)
        # Check if required columns exist
        required_columns = ['download_speed', 'upload_speed', 'latency']
        if not all(column in schools.columns for column in required_columns):
            return "Error: Required columns (download_speed, upload_speed, latency) are missing in the uploaded file.", 400

        # Add or update the Traffic_Label column
        schools['Traffic_Label'] = schools.apply(label_traffic, axis=1)
        schools.to_csv(schools_path, index=False)

        # Filter underserved schools
        underserved_schools = schools[schools['Traffic_Label'] == 'Low-Speed Traffic']

        # Determine optimal clusters and optimize tower locations
        n_clusters = determine_optimal_clusters(underserved_schools[['latitude', 'longitude']].values)
        optimized_centers = cluster_and_optimize(underserved_schools, n_clusters)

        # Prepare GeoJSON data
        tower_locations_df = pd.DataFrame(optimized_centers, columns=["latitude", "longitude"])
        geojson_data = df_to_geojson(tower_locations_df, "latitude", "longitude")
        geojson_path = os.path.join(RESULT_FOLDER, "potential_towers.geojson")
        with open(geojson_path, "w") as f:
            json.dump(geojson_data, f)

        # Render the map page with the GeoJSON file
        return render_template("map.html", geojson_file="results/potential_towers.geojson",
                               schools_data=json.dumps(schools_geojson),
        towers_data=json.dumps(cell_towers_geojson))

    return render_template("tower_location.html")

@app.route("/results/<path:filename>")
def serve_results(filename):
    return jsonify(json.load(open(os.path.join(RESULT_FOLDER, filename))))



# resource_allocation

from flask import Flask, request, jsonify
import joblib
import ctypes
import os
MODEL_PATH = 'traffic_model_whole_africa.pkl'
model = joblib.load(MODEL_PATH)

# Traffic shaping functions
def apply_traffic_rule(ip, bandwidth_limit):
    """Simulated traffic shaping rule for Windows (no tc available)."""
    try:
        # Windows does not support `tc` directly. Log the action instead.
        print(f"Simulating traffic shaping: IP={ip}, Bandwidth={bandwidth_limit} Mbps")
        return True
    except Exception as e:
        print(f"Error applying traffic rule: {e}")
        return False
@app.route('/classify', methods=['POST'])
def classify_traffic():
    """Classify traffic based on the input parameters."""
    data = request.json

    try:
        # Extract features from the request
        download_speed = data.get('download_speed')
        upload_speed = data.get('upload_speed')
        latency = data.get('latency')

        if download_speed is None or upload_speed is None or latency is None:
            return jsonify({"error": "Missing required features."}), 400

        # Predict traffic class
        features = [[download_speed, upload_speed, latency]]
        prediction = model.predict(features)[0]

        # Map numeric prediction back to class label
        label_map = {
            0: "Low-Speed Traffic",
            1: "Normal Traffic",
            2: "High-Speed Traffic",
            -1: "Unknown"
        }
        traffic_label = label_map.get(prediction, "Unknown")

        return jsonify({"traffic_label": traffic_label, "prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/prioritize', methods=['POST'])
def prioritize_traffic():
    """Apply traffic shaping based on traffic classification."""
    data = request.json

    try:
        # Extract features and IP from the request
        download_speed = data.get('download_speed')
        upload_speed = data.get('upload_speed')
        latency = data.get('latency')
        ip_address = data.get('ip_address', '0.0.0.0')  # Default to all traffic if not provided

        if download_speed is None or upload_speed is None or latency is None:
            return jsonify({"error": "Missing required features."}), 400

        # Predict traffic class
        features = [[download_speed, upload_speed, latency]]
        prediction = model.predict(features)[0]

        # Apply bandwidth prioritization based on class
        if prediction == 2:  # High-Speed Traffic
            bandwidth_limit = 100  # Full bandwidth (e.g., 100 Mbps)
        elif prediction == 1:  # Normal Traffic
            bandwidth_limit = 10  # Throttled (e.g., 10 Mbps)
        elif prediction == 0:  # Low-Speed Traffic
            bandwidth_limit = 1  # Highly throttled (e.g., 1 Mbps)
        else:
            return jsonify({"error": "Unknown traffic type, no rules applied."}), 400

        # Apply the traffic rule
        success = apply_traffic_rule(ip_address, bandwidth_limit)

        if not success:
            return jsonify({"error": "Failed to apply traffic rules."}), 500

        return jsonify({
            "message": "Traffic rules applied successfully.",
            "ip_address": ip_address,
            "bandwidth_limit": f"{bandwidth_limit} Mbps",
            "traffic_class": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#### procurment tasks 
# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename
# import os
# import fitz  # PyMuPDF for PDF extraction
# import requests
# from transformers import pipeline



# def extract_text_from_pdf(filepath):
#     """Extract text from a PDF file using PyMuPDF"""
#     try:
#         doc = fitz.open(filepath)
#         text = ""
#         for page in doc:
#             text += page.get_text()
#         return text
#     except Exception as e:
#         return str(e)

# def analyze_with_transformers(text):
#     """Analyze text using Hugging Face transformers pipeline"""
#     summarizer = pipeline("summarization")
#     summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
#     return summary[0]['summary_text']

# def evaluate_vendors(vendor_data):
#     """Evaluate vendors using Hugging Face sentiment analysis"""
#     sentiment_analyzer = pipeline("sentiment-analysis")
#     results = sentiment_analyzer(vendor_data)
#     return results

# @app.route('/procurment')
# def home():
#     return render_template("procurment.html")

# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)

#     # Extract text from the PDF
#     extracted_text = extract_text_from_pdf(filepath)

#     # Analyze text using transformers pipeline
#     analysis_result = analyze_with_transformers(extracted_text)

#     return jsonify({
#         "extracted_text": extracted_text,
#         "transformers_analysis": analysis_result
#     })

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     data = request.json.get('vendor_data', [])
#     if not data:
#         return jsonify({"error": "No vendor data provided"}), 400

#     evaluation_results = evaluate_vendors(data)
#     return jsonify({"evaluations": evaluation_results})
if __name__ == "__main__":
    app.run(debug=True)
