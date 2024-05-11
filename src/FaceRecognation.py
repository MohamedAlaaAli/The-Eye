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
        with open('../Classifier/pca_model.pkl', 'rb') as f:
            self.pca = pickle.load(f)
        self.projected_images = np.load("../Classifier/projected_images.npy")
        with open("Classifier\SVM_model.pkl", 'rb') as f:
            self.classifier = pickle.load(f)
    def preprocess_images(self):
        self.preprocessed_images = []
        labels = []
        for label, person_dir in enumerate(os.listdir(self.faces_dir)):
            person_path = os.path.join(self.faces_dir, person_dir)
            if os.path.isdir(person_path):
                for filename in os.listdir(person_path):
                    img_path = os.path.join(person_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (500, 500))
                        self.preprocessed_images.append(img.flatten())
                        labels.append(label)
                    else:
                        print(f"Warning: Unable to read image '{img_path}'. Skipping...")
        if not self.preprocessed_images:
            print("Error: No valid images found in the specified directory.")
            return None, None
        self.preprocessed_images = np.array(self.preprocessed_images)
        self.labels = np.array(self.labels)

    def train_classifier(self, labels):
        self.pca = PCA(n_components=4)  # Adjust the number of components as needed
        self.pca.fit(self.preprocessed_images)
        eigenfaces = self.pca.components_

        projected_images = self.pca.transform(self.preprocessed_images)
        # Load PCA object from file

        svm_classifier = SVC(kernel='rbf')  # Use SVM with a radial basis function kernel
        svm_classifier.fit(projected_images, labels)

    def recognize_faces(self, test_image):
        try:
            # Resize and preprocess the test image
            preprocessed_test_image = cv2.resize(test_image, (500, 500)).flatten().reshape(1, -1)

            # Project the test image onto the eigenfaces
            projected_test_image = self.pca.transform(preprocessed_test_image)

            # Predict the label using the trained classifier
            predicted_label = self.classifier.predict(projected_test_image)
            return predicted_label

        except Exception as e:
            print(f"An error occurred: {e}")
            return None




# Example usage:
face_recognition = EigenFaceRecognition()
face_recognition.load_pca_and_classifier()
results = []
detector = OfflineFaceDetection()
img_path = "../IMG_4738.jpeg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector.detect_faces(img, 1.3, 10, 100)
cropped_face = detector.crop_faces(gray, faces, "..", 1)
for face in cropped_face:
    # Test with a new image
    predicted_label = face_recognition.recognize_faces(face)
    results.append(predicted_label)
print(results)

# Assuming you have test directories for each person stored in a list test_dirs

# Example usage:
# face_recognition = EigenFaceRecognition(faces_dir, faces_dir)
# preprocessed_images, labels = face_recognition.preprocess_images()
# pca, classifier = face_recognition.train_classifier(preprocessed_images, labels)
#
# # Load test data for each person from their respective directories
# test_data_list, test_labels_list = face_recognition.load_test_data()
#
# # Visualize ROC curves for each person
# curves = face_recognition.calculate_roc_curves_for_classes(pca, classifier, test_data_list, test_labels_list)
# face_recognition.plot_roc_curves(curves)
