import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from FaceDetection import OfflineFaceDetection
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle

class EigenFaceRecognition:
    def __init__(self, faces_dir=None):
        self.faces_dir = faces_dir
        self.pca = None
        self.classifier = None
        self.projected_images = None
        self.preprocessed_images = None
        self.labels = None

    def load_pca_and_classifier(self):
        """
        Load the PCA model and the classifier from the specified files.

        This function loads the PCA model from the file '../Classifier/pca_model.pkl' and
        the classifier from the file 'Classifier/SVM_model.pkl'. It also loads the projected
        images from the file '../Classifier/projected_images.npy'.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        with open('Classifier/pca_model.pkl', 'rb') as f:
            self.pca = pickle.load(f)

        self.projected_images = np.load("Classifier/projected_images.npy")

        with open("Classifier/SVM_model.pkl", 'rb') as f:
            self.classifier = pickle.load(f)

            
    def preprocess_images(self):
        """
        Load images from the specified directory and preprocess them.

        This function loads the images from the directory specified in the constructor,
        and resizes them to have a fixed size of (500, 500) pixels. It also flattens
        the images and appends them to a list.

        Parameters:
            self (object): The instance of the class.

        Returns:
            preprocessed_images (numpy array): The preprocessed images.
            labels (numpy array): The labels of the images corresponding to their
                person.
        """
        self.preprocessed_images = []
        labels = []
        for label, person_dir in enumerate(os.listdir(self.faces_dir)):
            # Load the directory of the person
            person_path = os.path.join(self.faces_dir, person_dir)
            if os.path.isdir(person_path):
                # Iterate over the images in the directory
                for filename in os.listdir(person_path):
                    # Load the image
                    img_path = os.path.join(person_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # Check if the image was loaded successfully
                    if img is not None:
                        # Resize the image to have a fixed size of (500, 500) pixels
                        img = cv2.resize(img, (500, 500))
                        # Flatten the image
                        self.preprocessed_images.append(img.flatten())
                        # Append the label to the list of labels
                        labels.append(label)
                    else:
                        # Print a warning if the image was not loaded
                        print(f"Warning: Unable to read image '{img_path}'. Skipping...")
        # Check if any images were loaded
        if not self.preprocessed_images:
            # Print an error message if no images were loaded
            print("Error: No valid images found in the specified directory.")
            return None, None
        # Convert the list of images to a numpy array
        self.preprocessed_images = np.array(self.preprocessed_images)
        # Convert the list of labels to a numpy array
        self.labels = np.array(labels)


    def train_classifier(self, labels):
        """
        Train an SVM classifier using the eigenfaces as features.

        This function trains an SVM classifier on the eigenfaces extracted from the
        preprocessed images. The classifier is trained using the radial basis function
        kernel.

        Parameters:
            labels (numpy array): The labels of the images corresponding to their
                person.

        Returns:
            None
        """
        # Perform PCA on the preprocessed images
        self.pca = PCA(n_components=4)  # Adjust the number of components as needed
        self.pca.fit(self.preprocessed_images)

        # Extract the eigenfaces from the PCA model
        eigenfaces = self.pca.components_

        # Project the preprocessed images onto the eigenfaces
        projected_images = self.pca.transform(self.preprocessed_images)

        # Train an SVM classifier using the radial basis function kernel
        svm_classifier = SVC(kernel='rbf')  # Use SVM with a radial basis function kernel
        svm_classifier.fit(projected_images, labels)


    def recognize_faces(self, test_image):
        """
        Recognizes faces in a test image using a trained classifier.

        Parameters:
            test_image (numpy array): The test image to recognize faces in.

        Returns:
            predicted_label (int): The predicted label of the face in the test image.

        Raises:
            Exception: If an error occurs during the recognition process.
        """
        try:
            # Resize and preprocess the test image
            preprocessed_test_image = cv2.resize(test_image, (500, 500)).flatten().reshape(1, -1)

            # Project the test image onto the eigenfaces
            projected_test_image = self.pca.transform(preprocessed_test_image)

            # Predict the label using the trained classifier
            predicted_label = self.classifier.predict(projected_test_image)
            return predicted_label

        except Exception as e:
            print(f"An error occurred in recognition: {e}")
            return None


    def predict(self, test_image, scale_factor=1.1, min_neighbours=4, minSize=10):
        """
        Recognizes faces in a test image using a trained classifier.

        Parameters:
            test_image (numpy array): The test image to recognize faces in.

        Returns:
            results (list): A list of predicted labels of the faces in the test image.
        """
        results = []
        labels = {
            1: "Mohamed Alaa",
            2: "Mohamed Abdulaal",
            3: "Mandour",
            4: "Mohamed Ibrahim"
        }
        # Detect faces in the test image
        detector = OfflineFaceDetection()
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(
            test_image, scale_factor=scale_factor, min_neighbours=min_neighbours, minSize=minSize)

        # Test each face with the trained classifier
        for (x, y, w, h) in faces:
            # Crop the detected face
            face_gray = gray[y:y+h, x:x+w]

            # Test the face with the trained classifier
            predicted_labels = self.recognize_faces(face_gray)

            # If predicted_labels is an array, select the most common label
            if isinstance(predicted_labels, np.ndarray):
                predicted_label = np.argmax(np.bincount(predicted_labels))
            else:
                predicted_label = predicted_labels

            # Map the predicted label to its corresponding name
            label_name = labels.get(predicted_label, "Unknown")

            # Draw rectangle around the face
            cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 4)

            # Put label text above the face
            cv2.putText(test_image, label_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale= 0.9, color=(50, 255, 10), thickness= 3)

            # Store the predicted label
            results.append(label_name)

        return results, test_image


def main():
    # Example usage:
    face_recognition = EigenFaceRecognition()
    face_recognition.load_pca_and_classifier()
    img_path = "Images\Screenshot 2024-05-12 121822.png"
    img = cv2.imread(img_path)
    results, recognised_img = face_recognition.predict(img, 1.3, 10, 10)
    print(results)

    # Show the image with detected faces and labels in full screen mode
    # cv2.namedWindow("Detected Faces", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Detected Faces", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Show the image with detected faces and labels
    cv2.imshow("Detected Faces", recognised_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


