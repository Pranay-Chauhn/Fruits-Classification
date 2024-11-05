# Fruit Classification Project 🍎🍌🥭

## Overview
This project demonstrates a fruit classification model that predicts the type of fruit in an uploaded image. It combines deep learning, FastAPI, and a web-based frontend to create a complete end-to-end solution, allowing users to interact with a machine learning model directly from their browser.

### Project Structure
1. **Model Creation**: Trained a Convolutional Neural Network (CNN) in Google Colab to classify fruits.
2. **API Deployment**: Created a REST API using FastAPI to serve the model.
3. **Frontend**: Built a responsive HTML, CSS, and JavaScript-based web application for user interaction.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [API Development](#api-development)
- [Frontend Development](#frontend-development)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

---

## Installation

To run this project locally, follow these steps:

### Clone the Repository
```bash
git clone https://github.com/username/fruit-classification.git
cd fruit-classification
```

### Set Up Python Environment

To maintain compatibility, it's recommended to create a virtual environment with **Python 3.10**.

1. **Create a Virtual Environment**:
   ```bash
   python3.10 -m venv venv
   ```

2. **Activate the Virtual Environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### TensorFlow Serving
Ensure TensorFlow Serving is installed to serve the model.
- You can follow [TensorFlow Serving installation guide](https://www.tensorflow.org/tfx/guide/serving) for details.

---

## Dataset
The model is trained on a diverse dataset of fruits with **141 categories**. Each category folder contains images of specific fruits, helping the model generalize well across different classes.

## Model Training

Model training was conducted in **Google Colab** due to its GPU support. The training process included:

1. **Data Loading**: Loaded images, applied preprocessing techniques, and resized them to ensure consistency.
2. **Augmentation**: Enhanced model robustness through data augmentation techniques like rotation, flipping, and scaling.
3. **Model Architecture**: A Convolutional Neural Network (CNN) was implemented using **TensorFlow** and **Keras**.
4. **Evaluation**: Validated model accuracy on unseen data to ensure generalization.

*Colab Notebook link*: [https://github.com/Pranay-Chauhn/Fruits-Classification/blob/main/fruit_classification.ipynb](#)

### Model Export
The trained model was exported in `SavedModel` format to be used in TensorFlow Serving for deployment.

## API Development

The **FastAPI** framework was chosen for deploying the model and creating RESTful API endpoints.

### API Details

- **Endpoint**: `/predict` - Accepts image files as input and returns predicted fruit class and confidence score.
- **Input**: A POST request with an image file.
- **Output**: JSON response with the fruit class and confidence level.

#### Code Snippet
```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image, preprocess, and make a prediction
    ...
    return {"class": predicted_class, "confidence": confidence}
```

## Frontend Development

A simple, responsive web application was developed using **HTML**, **CSS**, and **JavaScript** for user interaction.

### Web App Features
- **File Upload**: Allows users to upload an image of a fruit.
- **Prediction Display**: Shows the predicted class and confidence score beside the uploaded image.
- **UI Design**: Styled with CSS to create a clean, user-friendly interface.

---

## Usage

### Running the Project

1. **Start TensorFlow Serving**: Ensure the model is running on the TensorFlow Serving port (e.g., `8501`).
2. **Run FastAPI Server**: Start the API server locally.
   ```bash
   uvicorn app:app --reload --port 8000
   ```
3. **Open the Web App**: Access `index.html` to interact with the model through the frontend.

---

## Future Improvements

- **Model Fine-Tuning**: Improve accuracy with additional data or hyperparameter tuning.
- **Frontend Enhancements**: Add more interactive UI elements, such as drag-and-drop image upload.
- **Deployment**: Consider deploying the API and web app on cloud services for public access.

---

Feel free to explore, fork, or contribute to this project. For any issues or questions, reach out via GitHub or LinkedIn.
