#The following python script takes a video(via webcam) as input and recognizes faces given in the known folder

import face_recognition
import os
import cv2

known_dir = "/home/user/Pictures_1/known"
tolerance = 0.5
frame_thickness = 3
font_thickness = 2
model = "CNN"
known_encodings = []
known_names = []

video = cv2.VideoCapture(0)

for file in os.listdir(known_dir):
  img = face_recognition.load_image_file(known_dir + '/' + file)
  img_enc = face_recognition.face_encodings(img)[0]
  known_encodings.append(img_enc)
  known_names.append(file.split('.')[0])
print("Processing unknown images")

while True:
    ret, image = video.read()
    locations = face_recognition.face_locations(image, model = model)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_encodings, face_encoding,tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            top_left = (face_location[3],face_location[0])
            bottom_right = (face_location[1],face_location[2])
            color = [0,255,0]
            cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), font_thickness)
    cv2.imshow(file, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
