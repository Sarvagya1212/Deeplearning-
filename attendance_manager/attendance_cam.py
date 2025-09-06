import cv2
import numpy as np
import face_recognition
import pickle
import time
from datetime import datetime, time as dt_time
from tensorflow.keras.models import model_from_json
import pandas as pd
import os

with open(r"C:\Users\sarva\Emotion_detection-main\attendance_manager\face_encodings.pkl", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

with open(r'C:\Users\sarva\Emotion_detection-main\Emotion_detectin\emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(r'C:\Users\sarva\Emotion_detection-main\Emotion_detectin\emotion_model_weights.h5')
emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

emotion_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

start_time = dt_time(9, 30)
end_time = dt_time(10, 0)

def is_within_time_window():
    current_time = datetime.now().time()
    return start_time <= current_time <= end_time

attendance = {}
detection_start_times = {}
min_detection_duration = 3  

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam attendance system. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    current_detected_names = set()

    for face_location, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = face_location
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        name = "Unknown"
        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_img = frame[top:bottom, left:right]
        if face_img.size == 0:
            continue  

        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized.astype('float32') / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)  
        face_input = np.expand_dims(face_input, axis=-1)      
        emotion_prediction = emotion_model.predict(face_input)
        emotion_index = np.argmax(emotion_prediction)
        emotion_label = emotion_labels[emotion_index]

        current_detected_names.add(name)

        if name != "Unknown" and is_within_time_window():
            current_time_sec = time.time()

            if name not in detection_start_times:
                detection_start_times[name] = current_time_sec

            elapsed_time = current_time_sec - detection_start_times[name]

            
            if elapsed_time >= min_detection_duration:

                if True: 
                    if name not in attendance:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        attendance[name] = {
                            "Time": timestamp,
                            "Status": "Present",
                            "Emotion": emotion_label
                        }
                        print(f"Attendance marked for {name} at {timestamp} with emotion {emotion_label}")

        label_text = f"{name}: {emotion_label}"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2)

    for student in list(detection_start_times.keys()):
        if student not in current_detected_names:
            del detection_start_times[student]


    cv2.imshow('Attendance & Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

all_students = set(known_face_names)
present_students = set(attendance.keys())
for student in all_students - present_students:
    attendance[student] = {"Time": "", "Status": "Absent", "Emotion": ""}

if attendance:
    attendance_df = pd.DataFrame.from_dict(attendance, orient='index')
    attendance_df.index.name = 'Student Name'
    attendance_df.to_csv('attendance_webcam.csv')
    print("Attendance saved to attendance_webcam.csv")
else:
    print("No attendance data to save.")
