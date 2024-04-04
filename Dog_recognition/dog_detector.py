import os
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Function to download and extract the Kaggle dataset
def download_dataset():
    os.system("kaggle competitions download -c dogs-vs-cats -p data")
    os.system("unzip data/dogs-vs-cats.zip -d data")

# Function to load and preprocess the user input image
import cv2

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize image to match the input size of the ResNet50 model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
    img_array = np.expand_dims(img, axis=0)
    return preprocess_input(img_array)

# Function to recognize if the image contains a dog
def recognize_dog(image_path):
    # Load the pre-trained ResNet50 model
    model = ResNet50(weights='imagenet')

    # Load and preprocess the user input image
    img_array = load_image(image_path)

    # Predict the image content
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=1)[0]

    # Check if the top prediction is a dog
    if decoded_preds[0][1] in ['beagle', 'bloodhound', 'bluetick', 'cocker_spaniel', 'golden_retriever', 'Irish_setter', 'Pembroke', 'pug', 'toy_terrier']:
        print("THIS IS A DOG")
    else:
        print("THIS IS NOT A DOG")

# Main function
def main():
    # Download the dataset if not already downloaded
    if not os.path.exists("data/train"):
        download_dataset()

    # Ask user for image path
    image_path = input("Enter the path to the image: ")

    # Check if the image exists
    if not os.path.exists(image_path):
        print("Invalid path to the image.")
        return

    # Recognize if the image contains a dog
    recognize_dog(image_path)

if __name__ == "__main__":
    main()
