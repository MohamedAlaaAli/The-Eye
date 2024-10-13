# The Eye


## Overview

The **The Eye** is a robust face detection and recognition application built using PyQt6. It leverages the power of OpenCV for offline and real-time face detection, along with EigenFace recognition for identifying faces. This tool offers a user-friendly interface with various customization options to fine-tune the detection and recognition processes.

## Features

- **Offline Face Detection**: Uses Haar cascades to detect faces in uploaded images.
- **Online Face Detection**: Real-time face detection using your computer's webcam.
- **Face Recognition**: Identify faces using the EigenFace method.
- **Customizable Parameters**: Adjust detection parameters such as window size, scale factor, and number of neighbors.
- **Interactive UI**: Easy-to-navigate interface with sliders, buttons, and real-time feedback.

## Eigenfaces Overview
Eigenfaces is a powerful face recognition technique that uses principal component analysis (PCA) to extract the most important features from a set of training faces. Here's a brief overview of how eigenfaces work:

1. Data Preparation: Collect a set of labeled face images for training. Preprocess the images by resizing them to a fixed size and converting them to grayscale.

2. PCA Application: Apply PCA to the preprocessed images to reduce their dimensionality and extract the most important features. The resulting eigenfaces represent the underlying structure in the data.

3. Training a Classifier: Use the eigenfaces as features to train a classifier, such as Support Vector Machines (SVM) or k-Nearest Neighbors (k-NN). The classifier learns to associate each face with its corresponding label.

4. Face Recognition: For a new, unseen face, preprocess it in the same way as the training images. Project the new face onto the eigenface space and compare its projection to the known faces using the trained classifier.

## Usage

### Main Window

Upon launching the application, you'll see the main window with the following sections:

- **Input Frame**: Upload images for offline face detection.
- **Output Frame**: Display the results of detection and recognition.
- **Technique Selection**: Choose between face detection and recognition.
- **Mode Selection**: Select offline or online mode for face detection.
- **Parameter Sliders**: Adjust window size, scale factor, and number of neighbors.

### Loading Images

1. Click on the **Import** button to browse and upload an image.
2. The selected image will be displayed in the input frame.

### Adjusting Parameters

Use the sliders to adjust:
- **Window Size**: Set the minimum window size for face detection.
- **Scale Factor**: Scale factor for the image pyramid.
- **Number of Neighbors**: Number of neighbors each candidate rectangle should have.

### Applying Techniques

1. **Face Detection**:
    - Choose **Detection**.
    - Select **Offline** or **Real Time** mode.
    - Click **Apply** to detect faces in the image or start real-time detection.

2. **Face Recognition**:
    - Choose **Recognition**.
    - Click **Apply** to identify faces in the uploaded image.

### Clearing and Resetting

- **Clear**: Clear the current image from the input and output frames.
- **Reset**: Reset the input image to its original state.

### Error Handling

If there is an error (e.g., camera not connected), an error message will be displayed.

## Code Overview

### Main Components

- `MainWindow`: The main application window class.
    - `init_ui()`: Initializes the UI components.
    - `apply_technique()`: Applies the selected technique (detection or recognition).
    - `apply_face_detection()`: Handles offline and online face detection.
    - `apply_face_recognition()`: Performs face recognition on the input image.
    - `browse_image()`: Allows the user to browse and select an image.
    - `update_label_text()`: Updates the slider labels with current values.
    - `check_camera_connected()`: Checks if the camera is connected.
    - `show_error_message()`: Displays error messages.

### Helper Classes

- `OfflineFaceDetection`: Handles offline face detection.
- `OnlineFaceDetection`: Manages real-time face detection using webcam.
- `EigenFaceRecognition`: Implements face recognition using the EigenFace method.

## Future Improvements

- Enhance face recognition accuracy with more advanced algorithms.
- Add support for multiple face detection and recognition models.
- Improve the UI for better user experience.

## Demo

<div align="center">
    <img src="331261735-e553cc26-2c62-4730-9081-370c14b567b8-ezgif.com-video-to-gif-converter.gif" alt="Demo of my project" />
</div>



