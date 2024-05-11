from FaceDetection import OfflineFaceDetection
import os
import cv2

face_detector = OfflineFaceDetection()
path = "../m_ibrahim"
for i, img in enumerate(os.listdir(path)):
    img_path = os.path.join(path, img)
    image = cv2.imread(img_path)
    faces = face_detector.detect_faces(image, scale_factor=1.1, min_neighbours=6, minSize=150)
    cropped_face = face_detector.crop_faces(image, faces, "../cropped_faces/m_ibrahim", i)