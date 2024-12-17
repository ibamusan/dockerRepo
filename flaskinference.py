import os
import whisper
from google.cloud import storage
from flask import Flask, request, jsonify

app = Flask(__name__)

# GCP Bucket details
MODEL_BUCKET_NAME = "vosyncore-transcription-dev"
MODEL_FOLDER = "whisper-model"
LOCAL_MODEL_DIR = "/tmp"  # Using /tmp instead of /app

# Global model variable
model = None

# Create the local directory for the model if it doesn't exist
if not os.path.exists(LOCAL_MODEL_DIR):
    os.makedirs(LOCAL_MODEL_DIR)

# Function to find and download the model dynamically
def find_and_download_model():
    print("Searching for Whisper model in GCP bucket...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(MODEL_BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=MODEL_FOLDER))

    for blob in blobs:
        if blob.name.endswith(".pt"):
            local_model_path = os.path.join(LOCAL_MODEL_DIR, os.path.basename(blob.name))
            print(f"Found model: {blob.name}. Downloading...")
            blob.download_to_filename(local_model_path)
            print(f"Model downloaded to {local_model_path}.")
            return local_model_path

    raise FileNotFoundError(f"No model file with '.pt' suffix found in GCP bucket '{MODEL_BUCKET_NAME}/{MODEL_FOLDER}'.")

# Load the Whisper model
def load_model():
    global model
    local_model_path = find_and_download_model()
    model = whisper.load_model(local_model_path)

@app.route("/health", methods=["GET"])
def health():
    return "Model is ready!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()
        input_bucket_name = data.get("input_bucket_name")
        output_bucket_name = data.get("output_bucket_name")
        input_audio_file_path = data.get("input_audio_file_path")
        output_folder = data.get("output_folder")

        # Validate input
        if not (input_bucket_name and output_bucket_name and input_audio_file_path and output_folder):
            return jsonify({"error": "Missing required fields in JSON request"}), 400

        # Download the audio file from GCS
        print(f"Downloading audio file from gs://{input_bucket_name}/{input_audio_file_path}...")
        storage_client = storage.Client()
        input_bucket = storage_client.bucket(input_bucket_name)
        audio_blob = input_bucket.blob(input_audio_file_path)
        local_audio_path = "/tmp/input_audio.wav"
        audio_blob.download_to_filename(local_audio_path)
        print(f"Audio file downloaded to {local_audio_path}.")

        # Perform transcription
        print("Starting transcription...")
        result = model.transcribe(local_audio_path)
        transcription = result["text"]
        print(f"Transcription completed: {transcription}")

        # Generate output file name based on input audio file name
        input_audio_filename = os.path.basename(input_audio_file_path)
        output_filename = os.path.splitext(input_audio_filename)[0] + "_transcribed.txt"
        output_gcs_path = f"{output_folder.rstrip('/')}/{output_filename}"
        local_output_path = f"/tmp/{output_filename}"

        # Save transcription to a text file
        with open(local_output_path, "w") as f:
            f.write(transcription)

        print(f"Uploading transcription to gs://{output_bucket_name}/{output_gcs_path}...")
        output_bucket = storage_client.bucket(output_bucket_name)
        output_blob = output_bucket.blob(output_gcs_path)
        output_blob.upload_from_filename(local_output_path)
        print("Transcription uploaded successfully.")

        return jsonify({"transcription_gcs_path": f"gs://{output_bucket_name}/{output_gcs_path}"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_model()  # Load model during startup
    app.run(host="0.0.0.0", port=8081)
