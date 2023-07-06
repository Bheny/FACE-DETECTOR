import cv2
import openface

# Load the pre-trained face detection model
face_detector_model = "models/shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(face_detector_model)

# Load the pre-trained face recognition model
face_recognition_model = "models/dlib_face_recognition_resnet_model_v1.dat"
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# Load the names and face encodings of known people
known_faces = {
    "Person 1": face_recognition.load_image_file("known_faces/person1.jpg"),
    "Person 2": face_recognition.load_image_file("known_faces/person2.jpg"),
    "Person 3": face_recognition.load_image_file("known_faces/person3.jpg"),
}
known_encodings = {}
for name, image in known_faces.items():
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    known_encodings[name] = face_encodings[0]

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    # Draw rectangles around the detected faces and label them
    for face in faces:
        # Extract face landmarks
        landmarks = landmark_predictor(gray, face)

        # Encode the face
        face_encoding = face_encoder.compute_face_descriptor(gray, landmarks)

        # Compare the face encoding with known encodings
        matches = face_recognition.compare_faces(list(known_encodings.values()), face_encoding)

        # Find the best match
        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = list(known_encodings.keys())[match_index]

        # Draw a rectangle around the face
        (x, y, w, h) = (face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put the label on the image
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
