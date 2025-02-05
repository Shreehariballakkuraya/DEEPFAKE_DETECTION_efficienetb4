import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from datetime import datetime
from io import BytesIO, StringIO
import csv
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Initialize Flask app
app = Flask(__name__)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the model class
class EfficientNetB4Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetB4Classifier, self).__init__()
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Load the trained model
model = EfficientNetB4Classifier()
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Prediction function for a single image
def predict_image(image):
    model.eval()  # Ensure model is in eval mode
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return "celeb-real" if predicted.item() == 0 else "celeb-synthesis", confidence.item()


# Prediction function for a video
def predict_video(video_path, frame_skip=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    # Create unique directory for this upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("static", "processed", timestamp)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    predictions = []
    confidences = []
    processed_frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip == 0:  # Process frames based on frame_skip value
            # Save original frame
            frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            # Convert to PIL Image for prediction
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prediction, confidence = predict_image(pil_image)
            predictions.append(prediction)
            confidences.append(confidence)

            processed_frames.append({
                'frame_number': frame_count,
                'frame_path': os.path.relpath(frame_path, 'static'),
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': f"{frame_count / fps:.1f}s"
            })

    cap.release()

    # Calculate metrics
    real_count = predictions.count("celeb-real")
    fake_count = predictions.count("celeb-synthesis")
    total_processed = len(predictions)

    if total_processed == 0:
        return {
            'prediction': "No frames processed",
            'confidence': 0,
            'real_count': 0,
            'fake_count': 0,
            'processed_frames': [],
            'output_dir': output_dir,
            'total_frames': total_frames,
            'processed_frames_count': 0,
            'fps': fps,
            'frame_skip': frame_skip
        }

    # Calculate weighted prediction based on confidence scores
    real_confidence_sum = sum(conf for pred, conf in zip(predictions, confidences) if pred == "celeb-real")
    fake_confidence_sum = sum(conf for pred, conf in zip(predictions, confidences) if pred == "celeb-synthesis")

    final_prediction = "real" if real_confidence_sum > fake_confidence_sum else "fake"
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    return {
        'prediction': final_prediction,
        'confidence': avg_confidence * 100,
        'real_count': real_count,
        'fake_count': fake_count,
        'processed_frames': processed_frames,
        'output_dir': output_dir,
        'total_frames': total_frames,
        'processed_frames_count': total_processed,
        'fps': fps,
        'frame_skip': frame_skip
    }


# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    if file:
        # Create temporary directory for uploads if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        # Save uploaded file
        temp_video_path = os.path.join("uploads", file.filename)
        file.save(temp_video_path)

        try:
            # Get frame skip value from form
            frame_skip = int(request.form.get('frame_skip', 15))

            # Process video and get results
            results = predict_video(temp_video_path, frame_skip=frame_skip)

            # Clean up
            os.remove(temp_video_path)

            return render_template(
                "index.html",
                prediction=results['prediction'],
                confidence_score=f"{results['confidence']:.2f}",
                real_frames=results['real_count'],
                fake_frames=results['fake_count'],
                processed_frames=results['processed_frames'],
                total_frames=results['total_frames'],
                fps=results['fps'],
                processed_frames_count=results['processed_frames_count'],
                frame_skip=results['frame_skip']
            )
        except Exception as e:
            print(f"Error processing video: {e}")
            # Clean up on error
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return render_template(
                "index.html",
                error="Error processing video. Please try again with a different video file."
            )

    return redirect(url_for("index"))


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route("/export/<format>", methods=["POST"])
def export_results(format):
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "csv":
            return export_csv(data, timestamp)
        elif format == "pdf":
            return export_pdf(data, timestamp)
        else:
            return jsonify({"error": "Invalid format"}), 400

    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({"error": str(e)}), 500


def export_csv(data, timestamp):
    try:
        # Create a string buffer to write CSV data
        si = StringIO()
        writer = csv.writer(si)

        # Write headers
        writer.writerow([
            "Frame Number",
            "Timestamp",
            "Prediction",
            "Confidence (%)",
            "Frame Path"
        ])

        # Write frame data
        for frame in data['processed_frames']:
            writer.writerow([
                frame['frame_number'],
                frame['timestamp'],
                frame['prediction'],
                f"{frame['confidence'] * 100:.2f}",
                frame['frame_path']
            ])

        # Create the response
        output = si.getvalue()
        si.close()

        # Create a BytesIO object for the response
        bio = BytesIO()
        bio.write(output.encode('utf-8'))
        bio.seek(0)

        filename = f"deepfake-analysis-{timestamp}.csv"

        return send_file(
            bio,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        print(f"CSV export error: {e}")
        return jsonify({"error": str(e)}), 500


def export_pdf(data, timestamp):
    try:
        # Create a BytesIO object for the PDF
        buffer = BytesIO()

        # Create the PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(letter),
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Container for the 'Flowable' objects
        elements = []

        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )

        # Add title
        elements.append(Paragraph("Deepfake Detection Analysis Report", title_style))
        elements.append(Spacer(1, 12))

        # Add summary information
        summary_data = [
            ["Analysis Summary"],
            ["Total Frames", str(data['total_frames'])],
            ["Processed Frames", str(data['processed_frames_count'])],
            ["Final Prediction", data['prediction'].upper()],
            ["Confidence Score", f"{data['confidence']:.2f}%"],
            ["Real Frames", str(data['real_count'])],
            ["Fake Frames", str(data['fake_count'])],
            ["FPS", str(data['fps'])],
            ["Frame Skip", str(data['frame_skip'])]
        ]

        summary_table = Table(summary_data, colWidths=[2 * inch, 2 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(summary_table)
        elements.append(Spacer(1, 20))

        # Add frame analysis table
        elements.append(Paragraph("Frame Analysis", styles['Heading2']))
        elements.append(Spacer(1, 12))

        frame_data = [["Frame #", "Timestamp", "Prediction", "Confidence"]]
        for frame in data['processed_frames']:
            frame_data.append([
                str(frame['frame_number']),
                frame['timestamp'],
                frame['prediction'],
                f"{frame['confidence'] * 100:.2f}%"
            ])

        frame_table = Table(frame_data, colWidths=[1.5 * inch, 1.5 * inch, 2 * inch, 1.5 * inch])
        frame_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white])
        ]))

        elements.append(frame_table)

        # Build PDF document
        doc.build(elements)

        # Get the value of the BytesIO buffer
        pdf = buffer.getvalue()
        buffer.close()

        filename = f"deepfake-analysis-{timestamp}.pdf"

        # Create a new BytesIO object for the response
        response_buffer = BytesIO()
        response_buffer.write(pdf)
        response_buffer.seek(0)

        return send_file(
            response_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        print(f"PDF export error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static/processed", exist_ok=True)
    app.run(debug=True)
