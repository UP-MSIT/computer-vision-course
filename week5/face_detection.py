import cv2
import numpy as np
import os


def load_haar_cascade():
    # Check if the Haar Cascade file exists
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError("Haar Cascade file does not exist.")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade


def detect_faces(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def extract_faces(img, faces, new_width=100, new_height=150):
    face_images = []
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        resized_face = cv2.resize(face, (new_width, new_height))
        face_images.append(resized_face)
    return face_images


def create_face_collage(face_image, original_image, cols=5, new_width=100, new_height=150):
    rows = (len(face_image) + 1 + cols - 1) // cols
    collage = np.zeros((rows * new_height, cols * new_width, 3), dtype=np.uint8)

    # Resize the original image to fit in the first block
    resized_original = cv2.resize(original_image, (new_width, new_height))
    collage[0:new_height, 0:new_width] = resized_original

    for idx, face in enumerate(face_image):
        row = (idx + 1) // cols
        col = (idx + 1) % cols
        collage[row * new_height:(row + 1) * new_height,
        col * new_width:(col + 1) * new_width] = face

    return collage


def image_detection(img_path):
    face_cascade = load_haar_cascade()
    img = cv2.imread(img_path)
    faces = detect_faces(img, face_cascade)
    face_images = extract_faces(img, faces)

    if face_images:
        collage = create_face_collage(face_images, img)
        cv2.imshow("Face Detection Collage", collage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No faces detected:")


if __name__ == "__main__":
    image_path = "photos.jpg"
    image_detection(image_path)
