import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from imutils import paths
import dlib

class EigenFaceRecognition:
    def __init__(self, data_dir, num_components=80):
        """
        Initialize the EigenFaceRecognition object.

        Parameters:
        data_dir (str): Path to the directory containing the dataset of images.
        num_components (int): Number of principal components to retain for PCA.
        """
        self.data_dir = data_dir
        self.num_components = num_components
        self.labels = []
        self.face_data = []
        self.names = {}
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def prepare_dataset(self):
        """
        Load images from the dataset directory and prepare the data for training.
        """
        image_paths = list(paths.list_images(self.data_dir))
        for i, image_path in enumerate(image_paths):
            name = os.path.basename(os.path.dirname(image_path))
            if name not in self.names:
                self.names[name] = len(self.names)
            label = self.names[name]
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = self.align_face(img)
            self.labels.append(label)
            self.face_data.append(img)

    def align_face(self, img):
        """
        Detect and align the face in the image.

        Parameters:
        img (numpy.ndarray): Input image.

        Returns:
        numpy.ndarray: Aligned face image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        if len(rects) > 0:
            shape = self.predictor(gray, rects[0])
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            left_eye_center = left_eye.mean(axis=0).astype("int")
            right_eye_center = right_eye.mean(axis=0).astype("int")
            dY = right_eye_center[1] - left_eye_center[1]
            dX = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            desired_right_eye_x = 1 - 0.35
            desired_dist = desired_right_eye_x - 0.65
            desired_dist *= 150
            eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
            tX = 0
            tY = 0
            M[0, 2] += (tX - eyes_center[0])
            M[1, 2] += (tY - eyes_center[1])
            (w, h) = (150, 150)
            output = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC)
            return output
        return img

    def train(self):
        """
        Train the face recognition model using Eigenfaces.
        """
        self.prepare_dataset()
        self.face_data = np.array(self.face_data)
        self.labels = np.array(self.labels)

        # Perform PCA
        pca = PCA(n_components=self.num_components)
        pca.fit(self.face_data)

        # Project face data to PCA subspace
        self.projected_data = pca.transform(self.face_data)

        # Train SVM classifier
        self.svm = SVC(kernel='linear', probability=True)
        self.svm.fit(self.projected_data, self.labels)

    def predict(self, test_img_path):
        """
        Predict the identity of a face in a test image.

        Parameters:
        test_img_path (str): Path to the test image.

        Returns:
        str: Name of the predicted person.
        """
        test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        test_img = self.align_face(test_img)
        test_img = test_img.reshape(1, -1)

        # Project test image to PCA subspace
        pca = PCA(n_components=self.num_components)
        pca.fit(self.face_data)
        projected_test_img = pca.transform(test_img)

        # Predict using SVM classifier
        prediction = self.svm.predict(projected_test_img)

        # Map prediction to name
        for name, label in self.names.items():
            if label == prediction[0]:
                predicted_name = name
                break
        return predicted_name

# Example usage:
if __name__ == "__main__":
    data_dir = 'dataset'  # Path to the directory containing the dataset of images
    test_img_path = 'test_image.jpg'  # Path to the test image
    eigen_face_recognition = EigenFaceRecognition(data_dir)
    eigen_face_recognition.train()
    predicted_name = eigen_face_recognition.predict(test_img_path)
    print("Predicted Name:", predicted_name)
