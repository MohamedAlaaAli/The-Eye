import cv2

class OfflineFaceDetection:
    def __init__(self, image = None, cascade_path = "Classifier/haarcascade_frontalface_default.xml"):
        """
        Initialize the FaceDetection object with the path to the Haar cascade file.

        Parameters:
        cascade_path (str): Path to the Haar cascade XML file for face detection.
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.image = image


    def detect_faces(self, scale_factor= 1.1, min_neighbours = 5, minSize = 100):
        """
        Detects faces in an image using the Viola-Jones algorithm.
        https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework

        Args:
            scale_factor (float, optional): Scaling factor used to reduce the image size and detect faces at different scales. Defaults to 1.1.
            min_neighbours (int, optional): Minimum number of neighbors a candidate rectangle should have to retain it. Defaults to 5.

        Returns:
            list: A list of tuples containing the coordinates of the detected faces.
        """
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbours, minSize=(minSize, minSize))
        return faces


    def draw_faces(self, faces, output_path=False):
        """
        Draw rectangles around the detected faces and save or display the result.

        Parameters:
        faces (list): A list of tuples containing the coordinates of the detected faces.
        output_path (str, optional): Path to save the output image with rectangles drawn around the faces. Defaults to False.

        Returns:
        segmented_image: The image with rectangles drawn around the detected faces.
        """
        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            segmented_image = cv2.rectangle(self.image, pt1=(x, y), pt2=(x+w, y+h), color= (0, 255, 0), thickness= 5)
        # Save or display the result
        if output_path:
            cv2.imwrite(output_path, segmented_image)
        print("Done")
        return segmented_image


class OnlineFaceDetection(OfflineFaceDetection):
    def __init__(self, cascade_path, image_viewport):
        """
        Initialize the OnlineFaceDetection object.

        Parameters:
        cascade_path (str): Path to the Haar cascade XML file for face detection.
        image_viewport (ImageViewport): Instance of the ImageViewport class for displaying frames.
        """
        super().__init__()
        self.image_viewport = image_viewport

    def run_face_detection(self):
        """
        Opens the default camera, captures frames, performs face detection, 
        displays the frame with detected faces, and releases the camera.
        No parameters. No return value.
        """
        # Open the default camera (usually webcam)
        cap = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Perform face detection and draw rectangles around faces
            frame_with_faces = self.draw_faces(frame)

            # Display the resulting frame with detected faces in ImageViewport
            self.image_viewport.set_image(None, camera_index=0)
            self.image_viewport.original_img = frame_with_faces  # Set the frame with detected faces
            self.image_viewport.update_display()

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()



def offline_detctetion():
    cascade_path = 'Classifier\haarcascade_frontalface_default.xml'  # Path to the Haar cascade XML file
    image_path = 'Images/3waad.ph (203).jpg'  # Path to the input image
    output_path = 'output_image.jpg'  # Path to save the output image with rectangles drawn around the faces
    image = cv2.imread(image_path)
    face_detector = OfflineFaceDetection(image, cascade_path)
    faces = face_detector.detect_faces( scale_factor= 1.1, min_neighbours = 5)
    face_detector.draw_faces( faces, output_path)

# Example usage:
if __name__ == "__main__":
    offline_detctetion()
