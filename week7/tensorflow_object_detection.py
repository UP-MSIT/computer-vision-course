import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load pre-trained model and lables
model = tf. keras.applications.MobileNetV2(weights='imagenet')
labels = tf.keras.applications.mobilenet_v2.decode_predictions

# Load the image
image_path = 'mixed_obj_02.jpg'
image = cv2.imread(image_path)

# Check if the image was successfully loaded
if image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Preprocess the image
    image_resized = cv2.resize(image, (224, 224))
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_resized)

    # Predict and decoce results
    predictions = model.predict(tf.expand_dims(image_array, axis=0))
    decoded_predictions = labels(predictions, top=10)  # Ttop = 10 is number of results needs
    print(decoded_predictions)

    # Convert BGR image to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a figure and axis for Matplotlib
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image_rgb)
    ax.axis('off')

    # set the annotation starting point one the right side
    annotation_start_x = image_rgb.shape[1] + 20
    annotation_start_y = 10

    # Annotation the image with the top prediction
    for i, pred in enumerate(decoded_predictions[0]):
        label, describtion, score = pred
        text = f"{describtion}: {score:.4f}"
        y = annotation_start_y + i * 75  # Row height gap size from each row of results
        ax.text(annotation_start_x, y, text, fontsize=12, color='black', backgroundcolor='white')

    # Adjust the plot to make space for the annotation on the right
    plt.subplots_adjust(left=0.05, right=0.75)

    # show the plot
    plt.show()
