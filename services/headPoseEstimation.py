import mediapipe as mp
import numpy as np
import cv2
import time
import random

# Constants
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
LANDMARK_INDICES = [33, 263, 1, 61, 291, 199]
SMILE_RATIO_THRESHOLD = 0.2
BLINK_DISTANCE_THRESHOLD = 6
FORWARD_HOLD_TIME = 3
CHALLENGE_TIME_LIMIT = 10
ANGLE_THRESHOLDS = {
    "Look Left": -10,
    "Look Right": 10,
    "Look Up": 10,
    "Look Down": -10
}
CHALLENGES = ["Look Left", "Look Right", "Look Up", "Look Down", "Smile", "Blink", "Forward"]
WINDOW_NAME = 'Head Pose Detection with Challenges'
FORWARD_IMAGE_FILENAME = 'forward_image.jpg'


class HeadPoseEstimator:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def run_estimation(self, cap):

        completed_challenges = []
        current_challenge = random.choice(CHALLENGES)
        challenge_start_time = time.time()
        forward_image = None
        forward_start_time = None

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Check challenge time limit
            if time.time() - challenge_start_time > CHALLENGE_TIME_LIMIT:
                cap.release()
                return False, None

            img_h, img_w = image.shape[:2]
            result = self._process_frame(image, current_challenge)

            # Always draw challenge and time
            self._draw_ui(image, current_challenge, challenge_start_time, img_h=img_h, img_w=img_w)

            if result is not None:
                angles, text, smile_detected, blink_detected = result

                # Handle Forward challenge
                if current_challenge == "Forward" and text == "Forward":
                    if forward_start_time is None:
                        forward_start_time = time.time()
                    elif time.time() - forward_start_time >= FORWARD_HOLD_TIME:
                        forward_image = image.copy()
                        self._draw_completion_message(image, img_h)
                        cv2.imshow(WINDOW_NAME, image)
                        cv2.waitKey(2000)
                        completed_challenges.append(current_challenge)
                        forward_start_time = None
                        if len(completed_challenges) < len(CHALLENGES):
                            current_challenge = self._select_new_challenge(completed_challenges)
                            challenge_start_time = time.time()
                        else:
                            cap.release()
                            if forward_image is not None:
                                cv2.imwrite(FORWARD_IMAGE_FILENAME, forward_image)
                            return True, forward_image
                elif current_challenge != "Forward":
                    forward_start_time = None

                # Check challenge completion
                if self._is_challenge_completed(current_challenge, text, smile_detected, blink_detected):
                    if current_challenge not in completed_challenges:
                        completed_challenges.append(current_challenge)
                        if len(completed_challenges) < len(CHALLENGES):
                            self._draw_completion_message(image, img_h)
                            cv2.imshow(WINDOW_NAME, image)
                            cv2.waitKey(5000)
                            current_challenge = self._select_new_challenge(completed_challenges)
                            challenge_start_time = time.time()
                        else:
                            cap.release()
                            if forward_image is not None:
                                cv2.imwrite(FORWARD_IMAGE_FILENAME, forward_image)
                                print(f"Forward image saved as '{FORWARD_IMAGE_FILENAME}'")
                            return True, forward_image

                # Draw angles and text
                self._draw_ui(image, current_challenge, challenge_start_time, angles=angles, text=text, img_h=img_h, img_w=img_w)

            cv2.imshow(WINDOW_NAME, image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        return False, None

    def _process_frame(self, image, current_challenge):

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in LANDMARK_INDICES:
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

                success_pnp, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                x_angle = angles[0] * 360
                y_angle = angles[1] * 360
                z_angle = angles[2] * 360

                smile_detected = self._detect_smile(face_landmarks, img_h, img_w)
                blink_detected = self._detect_blink(face_landmarks, img_h)

                text = self._get_current_text(current_challenge, x_angle, y_angle, smile_detected, blink_detected)

                # Draw nose direction line
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))
                cv2.line(image, p1, p2, (255, 0, 0), 3)

                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)

        return (x_angle, y_angle, z_angle), text, smile_detected, blink_detected

    def _detect_smile(self, face_landmarks, img_h, img_w):

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

        return vertical_distance / horizontal_distance > SMILE_RATIO_THRESHOLD

    def _detect_blink(self, face_landmarks, img_h):

        left_eye_top = face_landmarks.landmark[159]
        left_eye_bottom = face_landmarks.landmark[145]
        right_eye_top = face_landmarks.landmark[386]
        right_eye_bottom = face_landmarks.landmark[374]

        left_eye_distance = abs(left_eye_top.y - left_eye_bottom.y) * img_h
        right_eye_distance = abs(right_eye_top.y - right_eye_bottom.y) * img_h

        return left_eye_distance < BLINK_DISTANCE_THRESHOLD and right_eye_distance < BLINK_DISTANCE_THRESHOLD

    def _get_current_text(self, current_challenge, x_angle, y_angle, smile_detected, blink_detected):

        if current_challenge == "Smile":
            return "Smile" if smile_detected else "Not Smile"
        elif current_challenge == "Blink":
            return "Blink" if blink_detected else "Not Blink"
        else:
            if y_angle < ANGLE_THRESHOLDS.get("Look Left", -10):
                return "Look Left"
            elif y_angle > ANGLE_THRESHOLDS.get("Look Right", 10):
                return "Look Right"
            elif x_angle < ANGLE_THRESHOLDS.get("Look Down", -10):
                return "Look Down"
            elif x_angle > ANGLE_THRESHOLDS.get("Look Up", 10):
                return "Look Up"
            else:
                return "Forward"

    def _is_challenge_completed(self, current_challenge, text, smile_detected, blink_detected):

        return ((current_challenge == "Look Left" and text == "Look Left") or
                (current_challenge == "Look Right" and text == "Look Right") or
                (current_challenge == "Look Up" and text == "Look Up") or
                (current_challenge == "Look Down" and text == "Look Down") or
                (current_challenge == "Smile" and smile_detected) or
                (current_challenge == "Blink" and blink_detected))

    def _select_new_challenge(self, completed_challenges):

        return random.choice([c for c in CHALLENGES if c not in completed_challenges])

    def _draw_completion_message(self, image, img_h):

        cv2.putText(image, "completed!", (20, img_h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    def _draw_ui(self, image, current_challenge, challenge_start_time, angles=None, text=None, img_h=None, img_w=None):

        cv2.putText(image, f"Challenge: {current_challenge}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
        time_left = max(0, int(CHALLENGE_TIME_LIMIT - (time.time() - challenge_start_time)))
        cv2.putText(image, f"Time Left: {time_left}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if angles is not None:
            x_angle, y_angle, z_angle = angles
            cv2.putText(image, f"Pitch: {x_angle:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, f"Yaw: {y_angle:.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, f"Roll: {z_angle:.2f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if text is not None and img_h is not None:
            cv2.putText(image, text, (20, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


def HeadPoseEstimation(cap):

    estimator = HeadPoseEstimator()
    return estimator.run_estimation(cap)
