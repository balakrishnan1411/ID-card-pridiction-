import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')


webcam = cv2.VideoCapture(0)


saved_images = 0


while True:
    
    ret, frame = webcam.read()
    
    if not ret:
        print("Failed to capture frame from webcam")
        break
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    
    id_card_detected = False

    
    if len(faces) > 0:
        id_card_detected = True
        print("Faces detected! People are wearing ID cards.")
    else:
        print("No faces detected! People are not wearing ID cards.")

    
    if len(upper_bodies) > 0:
        id_card_detected = True
        print("Upper bodies detected! People are wearing ID cards around their neck.")
    else:
        print("No upper bodies detected! People are not wearing ID cards around their neck.")

    
    if id_card_detected:
        cv2.putText(frame, "ID Card Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "ID Card Not Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imwrite(f"not_wearing_id_{saved_images}.jpg", frame)
        saved_images += 1

    
    cv2.imshow("ID Card Detection", frame)
    
   
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()
