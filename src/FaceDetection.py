import cv2

class OfflineFaceDetection:
    def __init__(self, cascade_path = "Classifier/haarcascade_frontalface_default.xml"):
        """
        Initialize the FaceDetection object with the path to the Haar cascade file.

        Parameters:
        cascade_path (str): Path to the Haar cascade XML file for face detection.
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)


    def detect_faces(self, image, scale_factor= 1.1, min_neighbours = 5, minSize = 100):
        """
        Detects faces in an image using the Viola-Jones algorithm.
        https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework

        Args:
            scale_factor (float, optional): Scaling factor used to reduce the image size and detect faces at different scales. Defaults to 1.1.
            min_neighbours (int, optional): Minimum number of neighbors a candidate rectangle should have to retain it. Defaults to 5.

        Returns:
            list: A list of tuples containing the coordinates of the detected faces.
        """
        try:
            self.image = image
            gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbours, minSize=(minSize, minSize))
            return faces
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        

    def draw_faces(self, image,  faces, output_path=False):
        """
        Draw rectangles around the detected faces and save or display the result.

        Parameters:
        faces (list): A list of tuples containing the coordinates of the detected faces.
        output_path (str, optional): Path to save the output image with rectangles drawn around the faces. Defaults to False.

        Returns:
        segmented_image: The image with rectangles drawn around the detected faces.
        """
        try:
            # Initialize segmented_image
            segmented_image = image
            
            # Draw rectangles around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(segmented_image, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=5)
            
            # Save or display the result
            if output_path:
                cv2.imwrite(output_path, segmented_image)
                print(f"Image saved at {output_path}")
            
            return segmented_image
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


class OnlineFaceDetection(OfflineFaceDetection):
    def __init__(self, cascade_path):
        super(OnlineFaceDetection, self).__init__(cascade_path)
        self.is_running = False  # Flag to control the loop

    def run_face_detection(self, image_port):
        cap = cv2.VideoCapture(0)
        self.is_running = True  # Set the flag to indicate that face detection is running

        while self.is_running:  # Continue looping while the flag is True
            ret, frame = cap.read()

            faces = self.detect_faces(frame)
            frame_with_faces = self.draw_faces(frame, faces)
            image_port.set_frame(frame_with_faces)

            # Check for a condition to stop the loop
            # For example, you can add a key press event to stop face detection
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' key to quit
                break

        # Release the webcam and clean up
        cap.release()
        cv2.destroyAllWindows()

    def stop_face_detection(self):
        self.is_running = False  # Set the flag to False to stop the loop

            
            

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
