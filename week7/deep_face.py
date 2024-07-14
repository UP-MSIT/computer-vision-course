import cv2
from deepface import DeepFace
import os
import numpy as np
import pandas as pd

# Directory containing images of known individuals
know_face_dir = r'faces_known'
model_name = 'VGG-Face'

# Verify the directory exists
if not os.path.exists(know_face_dir):
    print(f"The directory {know_face_dir} does not exist.")
    raise FileNotFoundError(f'The directory {know_face_dir}')

# Print the current working directory and contents of the know_faces_dir
print('Current working directory: ', os.getcwd())
print('Know face directory: ', know_face_dir)
print('Contents of current working directory: ', os.listdir(know_face_dir))

# Load know faces and their names
know_faces = []
know_names = []

for filename in os.listdir(know_face_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = cv2.imread(os.path.join(know_face_dir, filename))
        know_faces.append(image_path)
        know_names.append(os.path.splitext(filename)[0])

# Preprocess know faces and create a data frame for them
face_df = pd.DataFrame(columns=["identity", "embedding"])
for img_path, name in zip(know_faces, know_names):
    embedding = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False)
    new_row = pd.DataFrame({"identity": [img_path], "embedding": [embedding[0]]})
    face_df = pd.concat([face_df, new_row], ignore_index=True)

# Open webcam capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    # Detect faces in the frame
    try:
        face_objs = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
        for face_obj in face_objs:
            face_location = face_obj["facial_area"]
            fx, fy, fw, fh = face_location['x'], face_location['y'], face_location['w'], face_location['h']
            face_roi = frame[fy:fy + fh, fx:fx + fw]

            # Convert face ROI to RGB
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Recognize the face
            embedding = DeepFace.represent(img_path=face_roi_rgb, model_name=model_name, enforce_detection=False)
            if embedding:
                distances = face_df['embedding'].apply(
                    lambda x: np.linalg.norm(np.array(x) - np.array(embedding[0]['embedding'])))
                min_distance_index = distances.idxmin()
                min_distance = distances[min_distance_index]

                if min_distance < 0.6:  # Threshold for face recognition
                    name = face_df.iloc[min_distance_index]["identity"]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            # Draw rectangle around the face and label it
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            cv2.putText(frame, name, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    except Exception as e:
        print(f"Error detecting face: {e}")

    # Display the resulting frame
    cv2.imshow('Live Face Tracking', frame)

    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
