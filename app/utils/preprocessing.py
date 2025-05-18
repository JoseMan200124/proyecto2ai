import cv2, mediapipe as mp, numpy as np

mp_hands = mp.solutions.hands

def extract_hand_landmarks(image_bgr: np.ndarray) -> np.ndarray | None:
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=1,
                        model_complexity=1,
                        min_detection_confidence=0.5) as hands:
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            return None
        coords = [(lm.x, lm.y, lm.z) for lm in res.multi_hand_landmarks[0].landmark]
        return np.array(coords).flatten()
