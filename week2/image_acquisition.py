import cv2
import matplotlib.pyplot as plt
from datetime import datetime


def adjust_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    lim = 255 - value

    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))

    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def capture_image(frame, description):
    filename = f"images/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{description}.jpg"

    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f'Image saved as {filename}')

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f'Captured image: {filename}')
    plt.show()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera is not opened.")

else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_adjusted = adjust_brightness(frame, value=50)

        cv2.imshow('Original', frame)
        cv2.imshow('Brightness', frame_adjusted)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            capture_image(frame, 'Original')
        elif key == ord('b'):
            capture_image(frame_adjusted, 'brightness_adjusted')
    cap.release()
    cv2.destroyAllWindows()