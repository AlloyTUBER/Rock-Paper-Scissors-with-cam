import pickle
import cv2
import mediapipe as mp
import numpy as np
import random

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
set_dict = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

game_result = ""
predicted_character = ""

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
        cv2.putText(frame, f'Gesture: {predicted_character}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

    if game_result:
        cv2.putText(frame, game_result, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, 'Press "e" to play, "q" to quit.', (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Rock Paper Sissors Game', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('e') and predicted_character != "":
 
        user_move = predicted_character
        comp_choice = random.randint(0, 2)
        comp_move = set_dict[comp_choice]

        if user_move == comp_move:
            game_result = f"Tie! Both chose {user_move}"
        elif (user_move == 'Rock' and comp_move == 'Sissors') or \
             (user_move == 'Paper' and comp_move == 'Rock') or \
             (user_move == 'Scissors' and comp_move == 'Paper'):
            game_result = f"You Win! {user_move} beats {comp_move}"
        else:
            game_result = f"You Lose! {comp_move} beats {user_move}"

cap.release()
cv2.destroyAllWindows()
