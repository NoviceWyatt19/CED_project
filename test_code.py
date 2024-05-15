
# it works in virtrual environment
# source my_venv/bin/activate
# when it want to done, input 'deactivate' in terminal

import cv2
import numpy as np   
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load face detection model and trained emotion recognition model
face_detection = cv2.CascadeClassifier('./sample/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('./sample/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Video capture using webcam
camera = cv2.VideoCapture(0)

while True:
    # Capture image from camera
    ret, frame = camera.read()
    
    # Convert color to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face detection in frame
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30))
    
    # Perform emotion recognition only when face is detected
    if len(faces) > 0:
        for face in faces:
            (fX, fY, fW, fH) = face
            roi_gray = gray[fY:fY + fH, fX:fX + fW]
            roi_gray = cv2.resize(roi_gray, (64, 64))  # Resize input to match model's expected input size
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            
            # Emotion predict
            preds = emotion_classifier.predict(roi_gray)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            
            # Assign labeling
            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
    
    # Display image ("Emotion Recognition")
    cv2.imshow('Emotion Recognition', frame)
    
    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clear program and close windows
camera.release()
cv2.destroyAllWindows()
