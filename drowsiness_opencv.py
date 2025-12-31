import cv2
import threading
from playsound import playsound

ALARM_SOUND = "H:/d/sounds/alert-sound.mp3"

face_cascade = cv2.CascadeClassifier(
    "H:/d/haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    "H:/d/haarcascade_eye.xml"
)

EYE_MISSING_FRAMES = 25   # ~1 sec (adjustable)
eye_missing_counter = 0
alarm_on = False

def play_alarm():
    playsound(ALARM_SOUND)

cap = cv2.VideoCapture(0)
print("🚗 Driver Drowsiness Detection Started (Improved Eye Logic)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = False

    for (x, y, w, h) in faces:
        # Focus only on upper half of face (where eyes are)
        roi_gray = gray[y:y + h//2, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        if len(eyes) > 0:
            eyes_detected = True

        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # ---- IMPROVED DROWSINESS LOGIC ----
    if len(faces) > 0 and not eyes_detected:
        eye_missing_counter += 1
    else:
        eye_missing_counter = 0
        alarm_on = False

    if eye_missing_counter >= EYE_MISSING_FRAMES:
        cv2.putText(frame, "DROWSINESS ALERT!", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        if not alarm_on:
            alarm_on = True
            threading.Thread(target=play_alarm, daemon=True).start()

    cv2.putText(frame, f"Eye-miss frames: {eye_missing_counter}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
