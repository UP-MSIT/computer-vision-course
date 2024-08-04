import cv2
import os

# Directory to save captured images
known_faces_dir = 'member_faces'

# Create the directory if it doesn't exist
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Prompt the user for the person's name
    person_name = input("Enter the name of the person (or 'q' to quit): ")
    if person_name.lower() == 'q':
        break

    # Directory to save the person's images
    person_dir = os.path.join(known_faces_dir, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    # Counter for image filenames
    img_counter = 0

    print("Press 'c' to capture an image, 'n' to start a new person, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow("Capture Images", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Capture and save the image
            img_name = os.path.join(person_dir, f"{person_name}_{img_counter}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Captured {img_name}")
            img_counter += 1
        elif key == ord('n'):
            break
        elif key == ord('q'):
            break

    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
