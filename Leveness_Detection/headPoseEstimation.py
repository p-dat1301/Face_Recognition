import mediapipe as mp
import numpy as np
import cv2
import time
import random

def HeadPoseActimation():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not accessible")
        return False, None

    # Chờ camera ổn định
    time.sleep(1) 

    # Challenge list
    challenges = ["Look Left", "Look Right", "Look Up", "Look Down", "Smile", "Blink", "Forward"]
    completed_Challege = []
    current_Challenge = random.choice(challenges)

    # Define thresholds for angles
    angle_threshold = {
        "Look Left": -20,
        "Look Right": 20,
        "Look Up": 20,
        "Look Down": -15
    }

    challenge_time_limit = 10
    challenge_start_time = time.time()

    forward_image = None
    forward_start_time = None

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            break

        # time_out check
        if time.time() - challenge_start_time > challenge_time_limit:
            cap.release()
            return False, None

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360  
                y = angles[1] * 360  
                z = angles[2] * 360

                # Hiển thị góc quay
                cv2.putText(image, f"Pitch: {x:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(image, f"Yaw: {y:.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(image, f"Roll: {z:.2f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Detect Smile
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]
                left_lip = face_landmarks.landmark[61]
                right_lip = face_landmarks.landmark[291]

                top_lip_y = int(top_lip.y * img_h)
                bottom_lip_y = int(bottom_lip.y * img_h)
                left_lip_x = int(left_lip.x * img_w)
                right_lip_x = int(right_lip.x * img_w)

                vertical_distance = abs(bottom_lip_y - top_lip_y)
                horizontal_distance = abs(right_lip_x - left_lip_x)

                smile_detected = vertical_distance / horizontal_distance > 0.2
                

                # Detect Blink
                left_eye_top = face_landmarks.landmark[159]
                left_eye_bottom = face_landmarks.landmark[145]
                right_eye_top = face_landmarks.landmark[386]
                right_eye_bottom = face_landmarks.landmark[374]

                left_eye_distance = abs(left_eye_top.y - left_eye_bottom.y) * img_h
                right_eye_distance = abs(right_eye_top.y - right_eye_bottom.y) * img_h

                blink_detected = left_eye_distance < 6.5 and right_eye_distance < 6.5
                    

                # Determine current state
                if current_Challenge == "Smile":
                    text = "Smile" if smile_detected else "Not Smile"
                elif current_Challenge == "Blink":
                    text = "Blink" if blink_detected else "Not Blink"
                else:
                    if y < angle_threshold.get("Look Left", -10):
                        text = "Look Left"
                    elif y > angle_threshold.get("Look Right", 10):
                        text = "Look Right"
                    elif x < angle_threshold.get("Look Down", -10):
                        text = "Look Down"
                    elif x > angle_threshold.get("Look Up", 10):
                        text = "Look Up"
                    else:
                        text = "Forward"

                # Handle Forward challenge
                if current_Challenge == "Forward" and text == "Forward":
                    if forward_start_time is None:
                        forward_start_time = time.time()
                    elif time.time() - forward_start_time >= 3:
                        forward_image = image.copy()
                        cv2.putText(image, "completed!", (20, img_h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                        cv2.imshow('Head Pose Detection with Challenges', image)
                        cv2.waitKey(2000)
                        completed_Challege.append(current_Challenge)
                        forward_start_time = None
                        if len(completed_Challege) < len(challenges):
                            current_Challenge = random.choice([c for c in challenges if c not in completed_Challege])
                            challenge_start_time = time.time()
                        else:
                            cap.release()
                            if forward_image is not None:
                                cv2.imwrite('forward_image.jpg', forward_image)
                                print("Forward image saved as 'forward_image.jpg'")
                            return True, forward_image
                elif current_Challenge != "Forward":
                    forward_start_time = None

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                # Display vector
                cv2.line(image, p1, p2, (255, 0, 0), 3)
                cv2.putText(image, text, (20, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                # Check Challenge
                if  (current_Challenge == "Look Left" and text == "Look Left") or \
                    (current_Challenge == "Look Right" and text == "Look Right") or \
                    (current_Challenge == "Look Up" and text == "Look Up") or \
                    (current_Challenge == "Look Down" and text == "Look Down") or \
                    (current_Challenge == "Smile" and smile_detected) or \
                    (current_Challenge == "Blink" and blink_detected):

                    if current_Challenge not in completed_Challege:
                        completed_Challege.append(current_Challenge)
                        if len(completed_Challege) < len(challenges):
                            cv2.putText(image, f"completed!", (20, img_h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                            cv2.imshow('Head Pose Detection with Challenges', image)
                            cv2.waitKey(5000)  # Thời gian chờ 5 giây
                            current_Challenge = random.choice([c for c in challenges if c not in completed_Challege])
                            challenge_start_time = time.time()
                        else:
                            cap.release()
                            if forward_image is not None:
                                cv2.imwrite('forward_image.jpg', forward_image)
                                print("Forward image saved as 'forward_image.jpg'")
                            return True, forward_image

                # Display text
                cv2.putText(image, f"Challenge: {current_Challenge}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
                cv2.putText(image, f"Time Left: {max(0, int(challenge_time_limit - (time.time() - challenge_start_time)))}s",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        cv2.imshow('Head Pose Detection with Challenges', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
