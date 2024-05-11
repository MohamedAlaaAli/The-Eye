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
    def __init__(self, faces_dir, test_dir):
        self.faces_dir = faces_dir
        self.test_dirs = test_dir

    def preprocess_images(self):
        preprocessed_images = []
        labels = []
        for label, person_dir in enumerate(os.listdir(self.faces_dir)):
            person_path = os.path.join(self.faces_dir, person_dir)
            if os.path.isdir(person_path):
                for filename in os.listdir(person_path):
                    img_path = os.path.join(person_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (500, 500))
                        preprocessed_images.append(img.flatten())
                        labels.append(label)
                    else:
                        print(f"Warning: Unable to read image '{img_path}'. Skipping...")
        if not preprocessed_images:
            print("Error: No valid images found in the specified directory.")
            return None, None
        return np.array(preprocessed_images), np.array(labels)

    def train_classifier(self, labels):
        pca = None
        # Load PCA object from file
        with open('../pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)
        projected_images = np.load("../projected_images.npy")

        classifier = SVC(kernel='rbf')  # Use SVM with a radial basis function kernel
        classifier.fit(projected_images, labels)
        return pca, classifier

    def recognize_faces(self, pca, classifier, test_image):
        try:
            # Resize and preprocess the test image
            preprocessed_test_image = cv2.resize(test_image, (500, 500)).flatten().reshape(1, -1)

            # Project the test image onto the eigenfaces
            projected_test_image = pca.transform(preprocessed_test_image)

            # Predict the label using the trained classifier
            predicted_label = classifier.predict(projected_test_image)
            return predicted_label

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def calculate_roc_curves_for_classes(self, pca, classifier, test_data, test_labels):
        roc_curves = {}

        # Project the test data onto the eigenfaces
        projected_test_data = pca.transform(test_data)
        test_labels = np.array(test_labels)
        # Iterate over each class
        for class_label in range(len(classifier.classes_)):
            # Create binary labels (1 for the target class, 0 for other classes)
            binary_labels = (test_labels == class_label).astype(int)

            # Get decision scores for the binary classification problem
            decision_scores = classifier.decision_function(projected_test_data)

            # Calculate ROC curve for the binary classification problem
            fpr, tpr, _ = roc_curve(binary_labels, decision_scores[:, class_label])

            # Calculate AUC score for the binary classification problem
            auc_score = auc(fpr, tpr)

            # Store the ROC curve and AUC score
            roc_curves[class_label] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}

        return roc_curves

    def load_test_data(self):
        test_data_list = []
        test_labels_list = []
        for label, person_dir in enumerate(os.listdir(self.test_dirs)):
            person_path = os.path.join(self.test_dirs, person_dir)
            print(person_path)
            for filename in os.listdir(person_path):
                img_path = os.path.join(person_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (500, 500))
                    test_data_list.append(img.flatten())
                    test_labels_list.append(label)
                else:
                    print(f"Warning: Unable to read image '{img_path}'. Skipping...")
            # test_data_list.append(np.array(test_images))
            # test_labels_list.append(np.array(labels).flatten())
        return test_data_list, test_labels_list

    def plot_roc_curves(self, roc_curves):
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
        names = ["Alaa", "elsayed", "Mandour", "M_ibrahim"]
        for class_label, curve_data in roc_curves.items():
            fpr = curve_data['fpr']
            tpr = curve_data['tpr']
            auc_score = curve_data['auc']
            plt.plot(fpr, tpr, label=f'{names[class_label]} (AUC = {auc_score:.2f})')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-class Classification')
        plt.legend()
        plt.grid(True)
        plt.show()




# Example usage:
faces_dir = '../cropped_faces'
face_recognition = EigenFaceRecognition(faces_dir, faces_dir)
preprocessed_images, labels = face_recognition.preprocess_images()
pca, classifier = face_recognition.train_classifier(labels)
results = []
detector = OfflineFaceDetection()
img_path = "../20240308_115108.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector.detect_faces(img, 1.3, 10, 100)
cropped_face = detector.crop_faces(gray, faces, "..", 1)
for face in cropped_face:
    # Test with a new image
    predicted_label = face_recognition.recognize_faces(pca, classifier, face)
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
