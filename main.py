from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QVBoxLayout, QFileDialog, QMessageBox, QInputDialog, QGraphicsDropShadowEffect
from PyQt6.QtGui import QIcon, QColor
import sys
import cv2
from src.ViewPort import ImageViewport
from src.FaceDetection import OnlineFaceDetection, OfflineFaceDetection
from src.FaceRecognation import EigenFaceRecognition

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.init_ui()
        self.cascade_path = "Classifier/haarcascade_frontalface_default.xml"
        self.offline_face_detector = OfflineFaceDetection(self.cascade_path)
        self.online_face_detector = OnlineFaceDetection(self.cascade_path)
        self.face_recognition = EigenFaceRecognition()
        self.detection_img = None

   
    def init_ui(self):
        """
        Initialize the UI by loading the UI page, setting the window title, loading UI elements, and checking a specific UI element.
        """
        # Load the UI Page
        self.ui = uic.loadUi('LightWindow.ui', self)
        self.setWindowTitle("Image Processing ToolBox")
        self.setWindowIcon(QIcon("icons/image-layer-svgrepo-com.png"))
        self.load_ui_elements()
        self.init_sliders()
        self.update_label_text()
        self.apply_shadows()
        self.ui.applyTech.clicked.connect(self.apply_technique)
        self.ui.detectionRadio.setChecked(True)
        self.ui.offlineRadio.setChecked(True)

        self.ui.detectionRadio.toggled.connect(self.detection_radioToggled)
        self.ui.offlineRadio.toggled.connect(self.mode_radioToggled)

        self.ui.windowSlider.valueChanged.connect(self.update_label_text)
        self.ui.neighboursSlider.valueChanged.connect(self.update_label_text)

        self.ui.resetButton.clicked.connect(self.reset)
        self.ui.clearButton.clicked.connect(self.clear_image)

    def apply_shadows(self):
        shadow_params = [
            (3, 3, 60, 20, self.ui.tech_frame),
            (3, 3, 60, 20, self.ui.mode_frame),
            (3, 3, 60, 20, self.ui.prop_frame),
            (3, 3, 60, 20, self.ui.input_frame),
            (5, 5, 60, 20, self.ui.out_frame),
        ]

        for params in shadow_params:
            self.add_shadow(*params)

    def add_shadow(self, x_offset, y_offset, alpha, blur, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setOffset(x_offset, y_offset)
        shadow.setBlurRadius(blur)
        shadow.setColor(QColor(0, 0, 0, alpha))
        widget.setGraphicsEffect(shadow)


    def load_ui_elements(self):
        """
        Load UI elements and set up event handlers.
        """
        # Initialize input and output port lists
        self.input_ports = []
        self.out_ports = []

        # Define lists of original UI view ports, output ports
        self.ui_view_ports = [self.ui.input1]

        self.ui_out_ports = [self.ui.output1]

        # Create image viewports for input ports and bind browse_image function to the event
        self.input_ports.extend([
            self.create_image_viewport(self.ui_view_ports[i], lambda event, index=i: self.browse_image(event, index))
            for i in range(1)])

        # Create image viewports for output ports
        self.out_ports.extend(
            [self.create_image_viewport(self.ui_out_ports[i], mouse_double_click_event_handler=None) for i in range(1)])
        
        # Initialize import buttons
        self.import_buttons = [self.ui.importButton]

        # Bind browse_image function to import buttons
        self.bind_import_buttons(self.import_buttons, self.browse_image)



    def bind_import_buttons(self, buttons, function):
        """
        Bind a function to a list of buttons.

        Args:
            buttons (list): List of buttons to bind the function to.
            function (callable): The function to bind to the buttons.

        Returns:
            None
        """
        for i, button in enumerate(buttons):
            button.clicked.connect(lambda event, index=i: function(event, index))


    def bind_buttons(self, buttons, function):
        """
        Bind a function to a list of buttons.

        Args:
            buttons (list): List of buttons to bind the function to.
            function (callable): The function to bind to the buttons.

        Returns:
            None
        """
        print(f"Binding buttons")
        for i, button in enumerate(buttons):
            print(f"Binding button {i}")
            button.clicked.connect(lambda index=i: function(index))


    def create_viewport(self, parent, viewport_class, mouse_double_click_event_handler=None):
        """
        Creates a viewport of the specified class and adds it to the specified parent widget.

        Args:
            parent: The parent widget to which the viewport will be added.
            viewport_class: The class of the viewport to be created.
            mouse_double_click_event_handler: The event handler function to be called when a mouse double-click event occurs (optional).

        Returns:
            The created viewport.

        """
        # Create a new instance of the viewport_class
        new_port = viewport_class(self)

        # Create a QVBoxLayout with parent as the parent widget
        layout = QVBoxLayout(parent)

        # Add the new_port to the layout
        layout.addWidget(new_port)

        # If a mouse_double_click_event_handler is provided, set it as the mouseDoubleClickEvent handler for new_port
        if mouse_double_click_event_handler:
            new_port.mouseDoubleClickEvent = mouse_double_click_event_handler

        # Return the new_port instance
        return new_port


    def create_image_viewport(self, parent, mouse_double_click_event_handler):
        """
        Creates an image viewport within the specified parent with the provided mouse double click event handler.
        """
        return self.create_viewport(parent, ImageViewport, mouse_double_click_event_handler)
    

    def init_sliders(self):
        """
        Initializes the sliders in the UI.

        Sets the range and initial value of the scaleFactorSlider, neighboursSlider, and windowSlider.

        """
        self.ui.windowSlider.setRange(10, 200)
        self.ui.windowSlider.setValue(10)

        self.ui.neighboursSlider.setRange(1, 30)
        self.ui.neighboursSlider.setValue(2)


    def mode_radioToggled(self):
        if self.ui.offlineRadio.isChecked():
            self.ui.input_frame.show()
            self.ui.prop_frame.show()
            self.ui.applyTech.show()
            self.input_ports[0].clear()
            self.out_ports[0].clear()

            if self.online_face_detector.is_running:
                print("Stopping online face detector")
                self.online_face_detector.stop_face_detection()
        
        else:
            self.ui.input_frame.hide()
            self.ui.prop_frame.hide()
            self.ui.applyTech.hide()
            self.apply_online_face_detection()


    def detection_radioToggled(self):
        if self.ui.detectionRadio.isChecked():
            self.ui.mode_frame.show()
            self.ui.applyTech.show()
    
        else:
            self.ui.input_frame.show()
            self.ui.applyTech.show()
            self.ui.prop_frame.show()
            self.ui.mode_frame.hide()
            

        if self.online_face_detector.is_running:
                print("Stopping online face detector")
                self.online_face_detector.stop_face_detection()

        self.ui.offlineRadio.setChecked(True)
        self.input_ports[0].clear()
        self.out_ports[0].clear()



    def update_label_text(self):
        """
        Updates the label text based on the current value of the sliders.

        This function is connected to the slider valueChanged signal,
        and is called whenever the value of a slider changes.
        It updates the text of the label next to the slider to display
        the current value of the slider.
        """

        window_min = self.ui.windowSlider.value()
        self.ui.window_val.setText(f"{window_min}")

        neighbours_num = self.ui.neighboursSlider.value()
        self.ui.neighbours_val.setText(f"{neighbours_num}")


    def clear_image(self):
        """
        Clear the specifed input and output ports.

        Args:
            index (int): The index of the port to clear.
        """
        self.input_ports[0].clear()
        self.out_ports[0].clear()


    def reset(self):
        """
        Resets the image at the specified index in the input_ports list.

        Args:
            event: The event triggering the image clearing.
            index (int): The index of the image to be cleared in the input_ports list.
        """
        self.input_ports[0].set_image(self.image_path)
        self.out_ports[0].set_image(self.image_path, grey_flag=True)
        

    def apply_technique(self):
        if self.ui.detectionRadio.isChecked():
            self.apply_face_detection()
        else:
            self.apply_face_recognition()


    def apply_face_detection(self):
        """
        Applies face detection based on the radio button selection (offline or online).
        """
        if self.ui.offlineRadio.isChecked():
            self.apply_offline_face_detection()
        else:
            self.apply_online_face_detection()


    def validate_parameter(self, lineEdit, parameter_name, param_type=int):
        """
        Validates the parameter entered by the user.

        Args:
            lineEdit (QLineEdit): The QLineEdit widget for the parameter.
            parameter (str): The name of the parameter being validated.
            param_type (type): The type of parameter to validate (int or float). Defaults to int.

        Returns:
            int or float or None: The valid parameter value entered by the user, or None if the user cancels the input dialog.
        """
        # parameter = None

        while True:
            try:
                parameter_ = param_type(lineEdit.text())
                if  parameter_ < 1 or parameter_ > 3:
                    raise ValueError
            except ValueError:
                self.show_error_message(f"{parameter_name} must be i range [1, 3] {'integer' if param_type == int else 'float'}.")

                # Prompt user to enter a different parameter
                if param_type == int:
                    parameter_, ok = QInputDialog.getInt(self, f"Enter {parameter_name}", "Please enter a positive integer:")
                else:
                    parameter_, ok = QInputDialog.getDouble(self, f"Enter {parameter_name}", "Please enter a positive float:")

                if ok:
                    lineEdit.setText(str(parameter_))
                    continue  # Retry with the new parameter
                else:
                    return None  # Return None if user cancels
            else:
                break  # Valid parameter, exit loop
        return parameter_


    def get_detection_parameters(self):
        """
        Retrieves the detection parameters for window size, scale factor, and number of neighbors.

        :return: A tuple containing the window size, scale factor, and number of neighbors.
        :rtype: tuple
        """
        window_min = int(self.ui.windowSlider.value())
        neighbours_min = int(self.ui.neighboursSlider.value())
        scale_factor = self.validate_parameter(self.ui.scaleFactor_val, "scale Factor", float)

        if not scale_factor:
            self.show_error_message("please enter Scale Factor ")
            scale_factor = 1.1

        return window_min, scale_factor, neighbours_min


    def apply_offline_face_detection(self):
        """
        Apply offline face detection to the input image.

        If the online face detector is running, it will be stopped.

        Retrieves the detection parameters for window size, scale factor, and number of neighbors
        and passes them to the offline face detector to detect faces.
        The detected faces are then drawn on the image and the output is displayed.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        # Stop the online face detector if it is running
        if self.detection_img is not None:
            if self.online_face_detector.is_running:
                print("Stopping online face detector")
                self.online_face_detector.stop_face_detection()

            # Get the detection parameters
            window_min, scale_factor, neighbours_min = self.get_detection_parameters()

            # Make a copy of the input image
            image = self.detection_img.copy()

            # Detect faces using the offline face detector
            faces = self.offline_face_detector.detect_faces(
                image, scale_factor=scale_factor, min_neighbours=neighbours_min, minSize=window_min)

            # Draw the detected faces on the image
            detected_image = self.offline_face_detector.draw_faces(image, faces)

            # Set the output image and display it
            self.out_ports[0].original_img = detected_image
            self.out_ports[0].update_display()


    def apply_face_recognition(self):
        """
        Apply face recognition to the input image.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        if self.detection_img is not None:

            window_min, scale_factor, neighbours_min = self.get_detection_parameters()
            image = self.detection_img.copy()

            self.face_recognition.load_pca_and_classifier()
            labels, recognised_img = self.face_recognition.predict(image, scale_factor, neighbours_min, window_min)
            print(labels)

            self.out_ports[0].original_img = recognised_img
            self.out_ports[0].update_display()

        

    def check_camera_connected(self):
        """
        Check if a camera is connected.

        It checks if the camera is successfully opened and returns a boolean value indicating whether the camera is connected or not.

        Returns:
            bool: True if the camera is connected, False otherwise.
        """
        # Try to access the first camera (index 0)
        cap = cv2.VideoCapture(0)

        # Check if the camera is opened
        connected = cap.isOpened()

        # Release the capture object
        cap.release()

        return connected


    def apply_online_face_detection(self):
        """
        Apply online face detection to the first output viewport.

        This function clears the current image in the output viewport and then runs the online face detection algorithm on the image. 
        The results of the face detection are displayed in the output viewport.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        if self.check_camera_connected():
            image_viewport = self.out_ports[0]
            image_viewport.clear()
            self.online_face_detector.run_face_detection(image_viewport)
        else:
            self.show_error_message("Camera not connected")
  

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec()

###################################################################################
#               Browse Image Function and Viewports controls                      #
###################################################################################
        

    def browse_image(self, event, index: int):
        """
        Browse for an image file and set it for the ImageViewport at the specified index.

        Args:
            event: The event that triggered the image browsing.
            index: The index of the ImageViewport to set the image for.
        """
        # Define the file filter for image selection
        file_filter = "Raw Data (*.png *.jpg *.jpeg *.jfif)"

        # Open a file dialog to select an image file
        self.image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', './', filter=file_filter)

        # Check if the image path is valid and the index is within the range of input ports
        if self.image_path and 0 <= index < len(self.input_ports):

            # Set the image for the last hybrid viewport
            input_port = self.input_ports[index]
            input_port.set_image(self.image_path)
            output_port = self.out_ports[index]
            output_port.set_image(self.image_path, grey_flag=True)
            self.detection_img = input_port.original_img.copy()



def main():
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()