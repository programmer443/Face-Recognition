import glob
import os

import cv2
import face_recognition as fg
import numpy as np

cur_dir = os.getcwd()
path = os.path.join(cur_dir, 'faces/')
list_of_files = [f for f in glob.glob(path + '*.jpg')]
number_files = len(list_of_files)
names = list_of_files.copy()
remove = cur_dir + '\\faces\\'
known_encoding = []
known_img_name = []

try:
    for i in range(number_files):
        globals()['image_{}'.format(i)] = fg.load_image_file(list_of_files[i])
        try:
            globals()['image_encoding_{}'.format(i)] = fg.face_encodings(globals()['image_{}'.format(i)])[0]
        except Exception as errors:
            print("[-] No Face Detected!")
            print("[-] Error is: " + str(errors))
        known_encoding.append(globals()['image_encoding_{}'.format(i)])
        # Create array of known names
        names[i] = names[i].replace(remove, "")
        names[i] = names[i].replace('.jpg', "")
        known_img_name.append(names[i])
except Exception as error:
    print("[-] Error is: " + str(error))

try:
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("[+] Start Streaming.......")
    while True:
        _, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        try:
            if process_this_frame:
                try:
                    face_locations = fg.face_locations(rgb_small_frame)
                    face_encodings = fg.face_encodings(rgb_small_frame, face_locations)
                    face_names = []
                except Exception as errors:
                    print("[-] Error is: " + str(errors))
                for face_encoding in face_encodings:
                    matches = fg.compare_faces(known_encoding, face_encoding)
                    name = "unknown"
                    # For image distance
                    face_distances = fg.face_distance(known_encoding, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_img_name[best_match_index]
                    face_names.append(name)
        except Exception as errors:
            print("[-] Error is: " + str(errors))
        process_this_frame = not process_this_frame
        # Display the names
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)
            # Input text label with a name below the face
            cv2.rectangle(frame, (left, bottom - 33), (right, bottom), (0, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (33, 184, 38), 1)
        # Live Cam And Hit Q to quit
        cv2.imshow('Face Recognition Application', frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    print("[-] ......End Streaming")
    cap.release()
    cv2.destroyAllWindows()
except Exception as errors:
    print("[-] Error is: " + str(errors))
    print("[-] End Streaming")
