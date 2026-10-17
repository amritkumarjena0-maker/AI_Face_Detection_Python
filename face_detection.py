import cv2

def main():
    # Load OpenCV's pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("Error: Could not load the face detection model.")
        return

    # Open the default webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open the camera.")
        return

    print("AI Face Detection Started")
    print("Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read camera frame.")
            break

        # Convert the camera image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around each detected face
        for index, (x, y, width, height) in enumerate(faces, start=1):
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Face {index}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Display the total number of detected faces
        cv2.putText(
            frame,
            f"Faces Detected: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("AI Face Detection - Press Q to Exit", frame)

        # Exit when Q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Camera closed successfully.")

if __name__ == "__main__":
    main()