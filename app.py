import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
from mailjet_rest import Client

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the trained YOLOv8 model
model = YOLO("yolo11n.pt")

# Mailjet API details
mailjet_api_key = "ed52b0e9c72904b58fb84f035f737cd9"
mailjet_api_secret = "83f3c1cdaa9640266a8c98da7d0e2666"
sender_email = "sushantawate2813@gmail.com"
receiver_email = "sushant.awate21@pccoepune.org"

# Function to send an email using Mailjet
def send_email(subject: str, body: str):
    mailjet = Client(auth=(mailjet_api_key, mailjet_api_secret), version='v3.1')
    data = {
        'Messages': [
            {
                "From": {
                    "Email": sender_email,
                    "Name": "Animal Intrusion Detection"
                },
                "To": [
                    {
                        "Email": receiver_email,
                        "Name": "Receiver"
                    }
                ],
                "Subject": subject,
                "TextPart": body,
            }
        ]
    }
    result = mailjet.send.create(data=data)
    return result

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Open the uploaded file
    image = Image.open(io.BytesIO(await file.read()))

    # Run inference on the image
    results = model(image)

    # Print detected classes and their confidence scores
    detected_classes = []
    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]  # Get class name
            detected_classes.append(class_name)

    print("Detected classes:", detected_classes)  # Check what classes are detected

    # Extract animal detections (set your own allowed classes)
    animal_detections = []
    allowed_classes = {"elephant", "wild_boar"}  # Modify this list as needed
    min_confidence = 0.5  # Confidence threshold to filter low-confidence detections

    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]  # Get class name
            if class_name in allowed_classes and box.conf >= min_confidence:  # Filter by class and confidence
                animal_detections.append({
                    "type": class_name,
                    "confidence": round(float(box.conf), 4),  # Round for readability
                    "bounding_box": [round(float(coord), 2) for coord in box.xyxy[0]]  # Format bbox
                })

    # If animals are detected, send an email notification
    if animal_detections:
        subject = "Animal Intrusion Detected!"
        body = f"Animals detected: {', '.join([d['type'] for d in animal_detections])}. Please take necessary action."
        send_email(subject, body)

    # Return only animal detections
    return JSONResponse(content={"animals_detected": animal_detections})

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use port from environment variable or default to 5000
    uvicorn.run(app, host="0.0.0.0", port=port)  # Start the server
