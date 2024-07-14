import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import requests

# Load the pre-trained model
model_url = 'nlpconnect/vit-gpt2-image-captioning'
model = VisionEncoderDecoderModel.from_pretrained(model_url)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_url)
tokenizer = AutoTokenizer.from_pretrained(model_url)


# Function to generate caption for an image
def generate_caption(image_path):
    # Open image
    image = Image.open(image_path).convert("RGB")

    # Preprocess image
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    # Generate caption
    output_ids = model.generate(pixel_values)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # path to the image
    image_path = "beach.jpg"  # Update this path to correct image path

    # Generate caption
    caption = generate_caption(image_path)
    print(caption)
