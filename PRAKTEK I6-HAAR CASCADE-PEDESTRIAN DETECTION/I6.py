import cv2

body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Pakai path lengkap video
video_path = r'H:\kuliah\Semester4\prak pcd\GO TO GITHUB\A1-F2\PRAKTEK I5-HAAR CASCADE-FACE AND EYE DETECTION\jalan.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Gagal membuka video:", video_path)
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Selesai membaca video atau gagal baca frame")
        break

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
