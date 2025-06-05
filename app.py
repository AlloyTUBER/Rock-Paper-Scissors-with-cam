import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import random
from PIL import Image

st.title("🖐 Rock Paper Scissors - Gesture Game")

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
set_dict = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

predicted_character = ""
game_result = ""

cap = cv2.VideoCapture(0)

if st.button("📷 Capture & Play"):
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image from webcam.")
    else:
        H, W, _ = frame.shape
        data_aux, x_, y_ = [], [], []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            user_move = predicted_character
            comp_choice = random.randint(0, 2)
            comp_move = set_dict[comp_choice]

            if user_move == comp_move:
                game_result = f"🤝 Tie! Both chose {user_move}"
            elif (user_move == 'Rock' and comp_move == 'Scissors') or \
                (user_move == 'Paper' and comp_move == 'Rock') or \
                (user_move == 'Scissors' and comp_move == 'Paper'):
                game_result = f"🎉 You Win! {user_move} beats {comp_move}"
            else:
                game_result = f"😢 You Lose! {comp_move} beats {user_move}"

            st.success(f"Gesture Detected: **{user_move}**")
            st.info(game_result)
        else:
            st.warning("No hand detected. Try again.")

        st.image(frame_rgb, channels="RGB", caption="Captured Frame")

cap.release()
