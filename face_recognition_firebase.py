from firebase_admin import credentials, initialize_app, storage, firestore
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
from glob import glob


# firebase 연동
cred = credentials.Certificate("key/raspberry-pi-e7bcf-firebase-adminsdk-u1dq6-441cdf9877.json")
initialize_app(cred, {'storageBucket': "raspberry-pi-e7bcf.appspot.com"})

# 스토리지 버킷
bucket = storage.bucket()

# 데이터베이스 연동
unknownPeople_db = firestore.client()
unknownPeople_ref = unknownPeople_db.collection(u'unknown_people').document(u'time_and_url')

# 인식하고 싶은 사람의 이미지 로드
# img_paths = ["images/jinho.png", "images/hodong.png", "images/iu.png", "images/boyoung.png"]
img_paths = glob("images/*.png")

# 얼굴 인코딩, 이름 저장
known_face_encodings = []
known_face_names = []

for img_path in img_paths:
    img = face_recognition.load_image_file(img_path)
    face_encoding = face_recognition.face_encodings(img)[0]
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
wait_ms = 1

while True:
    # 낯선 사람이 인식되지 않았을 경우 0.001초 딜레이를 줌
    wait_ms = 1
    
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
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # 얼굴 인식 O 
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
            # 얼굴 인식 X -> 낯선 사람 -> firebase에 데이터 저장
            else:
                # 현재 시간 저장
                now = datetime.now()
                time1 = now.strftime("%Y%m%d-%H%M%S")
                time2 = f"{now.year}년-{now.month}월-{now.day}일-{now.hour}시-{now.minute}분-{now.second}초"
                print(f"시간: {time2}\n낯선 사람 인식!!")
                
                # 로컬에 낯선 사람 이미지 저장
                cv2.imwrite(f"unknownPeople/{time1}.png", frame)
                print("Local saved")
                
                # storage에 낯선 사람 이미지 업로드
                fileName = f"unknownPeople/{time1}.png"
                blob = bucket.blob(fileName)
                blob.upload_from_filename(fileName)
                print("Storage saved")

                # 저정된 이미지 url 가져오기
                blob.make_public()
                img_url = blob.public_url
                
                # (현재시각: 저장된 이미지 url)과 같은 형식의 데이터를 Firestore database에 저장  
                if not unknownPeople_ref.get().to_dict():
                    unknownPeople_ref.set({time2 : img_url})
                else:
                    unknownPeople_ref.update({time2 : img_url})                                
                print("Datebase saved")
                print("Upload finish!\n")
                
                # 낯선 사람이 인식됐을 경우 2초의 딜레이를 줌 -> 너무 많은 사진이 찍히는 것을 방지
                wait_ms = 2000
                
            face_names.append(name) 
        
        
    process_this_frame = not process_this_frame
    
    # 결과 출력
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 프레임 사이즈를 1/4로 줄였었기 때문에, 4배를 해줘야 원래의 얼굴 위치가 된다.
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
    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
