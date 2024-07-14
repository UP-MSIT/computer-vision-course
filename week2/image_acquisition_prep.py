import cv2

image = cv2.imread('image.png')

if image is None: 
    print("Error: Image not found")
    exit()

image_gray = cv2.cvtColor(cv2.GaussianBlur(image, (5,5), 0), cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

image_clahe = clahe.apply(image_gray)

if len(image.shape) == 3:
    image_color_corrected = cv2.cvtColor(image_clahe, cv2.COLOR_GRAY2BGR)
else:
    image_color_corrected = image_clahe

edges = cv2.Canny(image_color_corrected, 50, 150)

cv2.imshow('Original', image)
cv2.imshow('Enhance', image_color_corrected)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()