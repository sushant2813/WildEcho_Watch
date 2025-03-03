import os
import cv2
import numpy as np
import pandas as pd  # Import pandas for Excel logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from mailjet_rest import Client
from datetime import datetime  # Import datetime for timestamps

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__, static_folder="frontend", static_url_path="/")
CORS(app)  # Enable CORS for frontend requests

# Initialize Roboflow Client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# Initialize Mailjet Client
mailjet = Client(auth=(os.getenv("MAILJET_API_KEY"), os.getenv("MAILJET_API_SECRET")), version='v3.1')

# Define allowed animals
ALLOWED_ANIMALS = {"Elephant", "Hyena", "Leopard", "Lion", "Wild Boar"}

# Excel file path
EXCEL_FILE = "detections.xlsx"

def log_to_excel(detected_animals):
    """Logs detected animals to an Excel file with Date and Time."""
    data = []
    for animal in detected_animals:
        data.append({
            "Animal": animal["type"],
            "Confidence (%)": animal["confidence"],
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Time": datetime.now().strftime("%H:%M:%S")
        })

    df = pd.DataFrame(data)

    # Append to existing Excel file or create a new one
    if os.path.exists(EXCEL_FILE):
        existing_df = pd.read_excel(EXCEL_FILE)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_excel(EXCEL_FILE, index=False)
    print("✅ Animal detections logged to Excel.")

def send_email(detected_animals):
    """Send an email notification when a high-confidence animal is detected."""
    sender_email = os.getenv("MAILJET_SENDER_EMAIL")
    receiver_email = os.getenv("MAILJET_RECEIVER_EMAIL")

    subject = "🚨 High Confidence Animal Detection Alert!"
    body = "The following animals were detected with high confidence:\n\n"
    for animal in detected_animals:
        body += f"🦁 Type: {animal['type']}, Confidence: {animal['confidence']}%\n"

    email_data = {
        'Messages': [{
            "From": {"Email": sender_email, "Name": "Animal Detector"},
            "To": [{"Email": receiver_email, "Name": "User"}],
            "Subject": subject,
            "TextPart": body
        }]
    }

    result = mailjet.send.create(data=email_data)
    print("📧 Mailjet Response:", result.json())  # Debugging
    return result.json()

@app.route("/")
def home():
    """Serve frontend (index.html) from Flask."""
    return send_from_directory("frontend", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload, prediction, logging, and email notification."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        image = request.files["file"]
        image_path = "temp.jpg"
        image.save(image_path)

        # Call Roboflow API
        result = client.run_workflow(
            workspace_name=os.getenv("ROBOFLOW_WORKSPACE"),
            workflow_id=os.getenv("ROBOFLOW_WORKFLOW"),
            images={"image": image_path},
            use_cache=True
        )

        print("🔍 Roboflow Response:", result)  # Debugging

        # Extract detected animals with confidence above 95%
        predictions = result[0].get("predictions", {}).get("predictions", [])
        detected_animals = [
            {"type": d["class"], "confidence": round(d["confidence"] * 100, 2)}
            for d in predictions if d["class"] in ALLOWED_ANIMALS and d["confidence"] * 100 >= 80
        ]

        # Log detections to Excel
        if detected_animals:
            log_to_excel(detected_animals)
            send_email(detected_animals)
            response_data = {"animals_detected": detected_animals}
        else:
            response_data = {"message": "No animals detected with high confidence."}

        print("✅ Formatted Response:", response_data)  # Debugging
        return jsonify(response_data)

    except Exception as e:
        print("❌ Error:", str(e))  # Log error
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
