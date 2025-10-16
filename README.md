# Landmark Image Classifier

This project uses a deep learning model to identify famous landmarks from static images. It leverages a pre-trained TensorFlow Lite model, trained on a Google Landmarks dataset, for efficient and accurate classification.

<img width="1600" height="474" alt="image" src="https://github.com/user-attachments/assets/6495b5c9-4848-45ea-ab99-a70abd79da07" />

## Features
-   **Image-Based Recognition**: Classifies landmarks from user-provided image files.
-   **Efficient Deep Learning Model**: Utilizes a lightweight TensorFlow Lite (`.tflite`) model for fast inference.
-   **Confidence Scoring**: Outputs the recognized landmark's name and the model's confidence score for the prediction.
-   **Extensive Landmark Coverage**: The model is capable of recognizing a wide variety of global landmarks.
-   **Extendable Design**: The core logic can be easily adapted for real-time classification from a live webcam feed.

## Technologies Used
-   **Python 3.x**
-   **OpenCV**: For loading, displaying, and processing images.
-   **TensorFlow Lite**: For running inference with the classification model.
-   **NumPy**: For numerical operations and data preprocessing.

## Project Workflow
1.  **Image Loading**: The script loads a specified image from a file path using OpenCV.
2.  **Image Preprocessing**: The image is resized to 321x321 pixels and normalized to match the input requirements of the TFLite model.
3.  **Inference**: The preprocessed image is passed to the TensorFlow Lite interpreter, which runs the landmark classification model.
4.  **Output Processing**: The model's output, a probability distribution over all possible landmarks, is processed to find the landmark with the highest confidence score.
5.  **Display Results**: The name of the predicted landmark and its confidence score are printed to the console and overlaid onto the image, which is then displayed in a window.

## Setup and Installation
Follow these steps to set up the project locally.

### 1. Clone the repository:
```bash
git clone [https://github.com/Mannan-15/LandmarkRecognition_Project.git](https://github.com/Mannan-15/LandmarkRecognition_Project.git)
cd LandmarkRecognition_Project
```

### 2. Get the Labels:
You can download the dataset from kaggle from the google landmark recognition contest.

### 3. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install the required libraries:
```bash
pip install -r requirements.txt
```

## ▶️ Usage
To run the landmark recognition on your own image:

1.  Place an image of a landmark (e.g., `eiffel_tower.jpg`) in the main project directory.
2.  Modify the script to point to your image file.
3.  Execute the main Python script from your terminal:
    ```bash
    python your_script_name.py
    ```
-   A window will appear showing your image with the predicted landmark and confidence score written on it.
-   The prediction will also be printed in the terminal.
