import face_recognition
from face_recognition import load_image_file, face_encodings
import cv2
import numpy as np
import os


# 인식하고 싶은 사람의 이미지 로드
img_paths = ["images/jinho.png"]

# 얼굴 인코딩, 이름 저장
known_face_encodings = []
known_face_names = []

for img_path in img_paths:
    img = load_image_file(img_path)
    face_encoding = face_encodings(img)[0]
    known_face_encodings.append(face_encoding)
    file_name = os.path.split(img_path)[-1][:-4]
    known_face_names.append(file_name)
    
    
# 노트북: 웹켐, 라즈베리파이: 카메라
cam = cv2.VideoCapture(0)

# 이미지 너비, 높이 설정
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 변수 initialize
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # 캡쳐
    ret, frame = cam.read()

    # 빠른 얼굴 인식 처리를 위해 프레임 사이즈를 1/4로 줄임
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # BGR -> RGB 변환
    rgb_small_frame = small_frame[:, :, ::-1]

    # 시간을 절약하기 위해 다른 모든 프레임만 처리함
    if process_this_frame:
        # 얼굴 탐지 -> 인코딩
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 인식된 얼굴이 있는지 확인
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # 얼굴 인식 O 
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # 결과 출력
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 탐지된 얼굴에 박스 그리기
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 탐지된 얼굴의 인식 결과 출력
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    # q 누르면 반복문 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()