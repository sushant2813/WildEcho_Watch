import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from mailjet_rest import Client

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__, static_folder="frontend")
CORS(app)  # Enable CORS for frontend requests

# Initialize Roboflow Client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# Initialize Mailjet Client
mailjet = Client(auth=(os.getenv("MAILJET_API_KEY"), os.getenv("MAILJET_API_SECRET")), version='v3.1')

# Define allowed animals
ALLOWED_ANIMALS = {
    0: "Elephant",
    1: "Hyena",
    2: "Leopard",
    3: "Lion",
    4: "Wild Boar"
}

def send_email(detected_animals):
    """Send an email notification when an animal is detected."""
    sender_email = os.getenv("MAILJET_SENDER_EMAIL")
    receiver_email = os.getenv("MAILJET_RECEIVER_EMAIL")
    
    # Prepare email content
    subject = "Animal Detection Alert!"
    body = "The following animals have been detected:\n\n"
    for animal in detected_animals:
        body += f"🦁 Type: {animal['type']}, Confidence: {animal['confidence']}%\n"

    # Mailjet API request payload
    email_data = {
        'Messages': [{
            "From": {"Email": sender_email, "Name": "Animal Detector"},
            "To": [{"Email": receiver_email, "Name": "User"}],
            "Subject": subject,
            "TextPart": body
        }]
    }

    # Send email via Mailjet
    result = mailjet.send.create(data=email_data)
    print("Mailjet Response:", result.json())  # Debugging
    return result.json()

@app.route("/")
def serve_frontend():
    """Serve the frontend HTML file."""
    return send_from_directory("frontend", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload, prediction, and email notification."""
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

        print("Roboflow Response:", result)  # Debugging

        # Extract detected animals
        predictions = result[0].get("predictions", {}).get("predictions", [])
        detected_animals = [
            {"type": d["class"], "confidence": round(d["confidence"] * 100, 2)}
            for d in predictions if d["class"] in ALLOWED_ANIMALS.values()
        ]

        # If animals are detected, send an email
        if detected_animals:
            send_email(detected_animals)

        response_data = {"animals_detected": detected_animals}
        print("Formatted Response:", response_data)  # Debugging

        return jsonify(response_data)

    except Exception as e:
        print("Error:", str(e))  # Log error
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
