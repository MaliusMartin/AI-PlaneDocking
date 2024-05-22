import cv2
from django.shortcuts import render
from torchvision import transforms
import os
import onnxruntime as rt
import numpy as np


def home(request):
  return render(request, 'avdgsApp/home.html')

# Load ONNX model
def load_plane_model():
  model_path = os.environ.get('MODEL_PATH', 'avdgsApp/models/plane.onnx')  # Default to 'models' folder
  model_path = os.path.join(os.getcwd(), model_path)
  model = rt.InferenceSession(model_path)
  return model

# Define necessary transformations
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize((640, 640)),
  transforms.ToTensor(),
])

def process_frame(frame, model):
  preprocessed_frame = transform(frame)
  processed_frame = preprocessed_frame.unsqueeze(0)
  input_data = processed_frame.numpy()
  output_data = model.run([], {model.get_inputs()[0].name: input_data})[0]
  processed_frame = processed_frame.squeeze(0).permute(1, 2, 0).numpy()
  return processed_frame

# Camera feed view
def camera_feed(request):
  # Load the model
  model = load_plane_model()

  # Initialize the camera
  camera = cv2.VideoCapture(0)
  while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
      break

    processed_frame = process_frame(frame, model)

    # Display the processed frame
    cv2.imshow('Camera Feed', processed_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

  camera.release()
  cv2.destroyAllWindows()

  return render(request, 'avdgsApp/camera_feed.html')
