import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from src.FaceDetection import OfflineFaceDetection
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Convert input to numpy array if it's not already
        X = np.array(X)
        
        # Compute mean of the data
        self.mean = np.mean(X, axis=0)
        
        # Center the data
        X_centered = X - self.mean
        
        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvectors based on eigenvalues
        eigenvectors = eigenvectors.T
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[sorted_indices]
        
        # Select the top n_components eigenvectors
        self.components = sorted_eigenvectors[:self.n_components]

    def transform(self, X):
        # Convert input to numpy array if it's not already
        X = np.array(X)
        
        # Center the data
        X_centered = X - self.mean
        
        # Project the data onto the principal components
        projected = np.dot(X_centered, self.components.T)
        
        return projected


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
            1: "M Alaa",
            2: "Elsayed",
            3: "Mandour",
            4: "M Ibrahim"
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
            cv2.putText(test_image, label_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1, color=(50, 255, 10), thickness= 3)

            # Store the predicted label
            results.append(label_name)

        return results, test_image

    
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

    
    def plot_roc_curves(self, roc_curves):
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
        names = ["Alaa", "Elsayed", "Mandour", "M_ibrahim"]
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


