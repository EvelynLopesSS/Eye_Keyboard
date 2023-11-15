import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import time
import pyttsx3

speech_engine = pyttsx3.init()
speech_engine.setProperty('voice', 'pt-br')
# Sons
tecla = pyglet.media.load("tecla.aac", streaming=False)
esquerda = pyglet.media.load("esquerda.aac", streaming=False)
direita = pyglet.media.load("direita.aac", streaming=False)
apagar = pyglet.media.load("apagar.aac", streaming=False)

cap = cv2.VideoCapture(0)
board = np.zeros((400, 1200), np.uint8)
board[:] = 255

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

keyboard_width = 1200
keyboard_height = 700

keyboard = np.zeros((keyboard_height, keyboard_width, 3), np.uint8)

# Dimensões da região da webcam
webcam_width = 400
webcam_height = 200

# Posição da região da webcam no teclado
webcam_x = (keyboard_width - webcam_width) // 2
webcam_y = (keyboard_height - webcam_height - 80) - 400

# Configurações das teclas
key_width = 100
key_height = 100
key_spacing_x = 20
key_spacing_y = 20

num_rows = 4
num_cols = 10

total_height_keys = num_rows * (key_height + key_spacing_y) - key_spacing_y

start_x = (keyboard_width - (num_cols * key_width + (num_cols - 1) * key_spacing_x)) // 2
start_y = (keyboard_height - total_height_keys) // 2 + 120

keys_set_1 = {
    0: "Q", 1: "W", 2: "E", 3: "R", 4: "T", 5: "Y", 6: "U", 7: "I", 8: "O", 9: "P",
    10: "A", 11: "S", 12: "D", 13: "F", 14: "G", 15: "H", 16: "J", 17: "K", 18: "L", 19: "Falar",
    20: "Z", 21: "X", 22: "C", 23: "V", 24: "B", 25: "N", 26: "M", 27: ",", 28: "Apagar", 29: "----"
}


def letter(letter_index, text, letter_light):
    x = start_x + (letter_index % num_cols) * (key_width + key_spacing_x)
    y = start_y + (letter_index // num_cols) * (key_height + key_spacing_y)

    th = 3
    if letter_light is True:
        key = np.zeros((key_height, key_width, 3), np.uint8)
        key[:] = (57, 255, 20)

    else:
        key = np.zeros((key_height, key_width, 3), np.uint8)
        key[:] = (138, 154, 91)

    
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    font_th = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((key_width - width_text) / 2)
    text_y = int((key_height + height_text) / 2)

    cv2.putText(key, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_th)

    keyboard[y:y + key_height, x:x + key_width] = key


def midpont(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


font = cv2.FONT_HERSHEY_SIMPLEX


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpont(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpont(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)
    right_eye_region = np.array([
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(37).x, landmarks.part(37).y),
        (landmarks.part(38).x, landmarks.part(38).y),
        (landmarks.part(39).x, landmarks.part(39).y),
        (landmarks.part(40).x, landmarks.part(40).y),
        (landmarks.part(41).x, landmarks.part(41).y)
    ], np.int32)

    
    cv2.polylines(frame, [right_eye_region], isClosed=True, color=(0, 0, 255), thickness=2)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [left_eye_region], 255)

    eye = cv2.bitwise_and(gray, gray, mask=mask)
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape

    left_side_threshold_eye = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold_eye)

    right_side_threshold_eye = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold_eye)

    #cv2.imshow(' Limite Direito', right_side_threshold_eye)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 4
    else:
        gaze_ratio = left_side_white / right_side_white

    return gaze_ratio



frames = 0
letter_index = 0
blinking_frames = 0
text = ""
keyboard_selected = "Esquerda"
last_keyboard_selected = 'Esquerda'
space_added = False
text_spoken = False
apagar_sound_playing = False
falar_sound_playing = False

while True:
    _, frame = cap.read()
    keyboard[:] = (0, 0, 0)
    frames += 1

    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    active_letter = keys_set_1[letter_index]
   
    if active_letter == "----" and not space_added:
        text += ' '
        tecla.play()
        space_added = True
    elif active_letter != "----":
        space_added = False
    if active_letter == "Apagar":
        cv2.putText(keyboard, 'Apagar', (webcam_x - 300, webcam_y + 80), font, 4, (0, 0, 255), thickness=3)
        
        if text and blinking_frames == 5: 
            text = text[:-1]
            if not apagar_sound_playing:
                apagar.play()
                apagar_sound_playing = True
    else:
        apagar_sound_playing = False
    if active_letter == "Falar" and not text_spoken and blinking_frames == 5: 
        
        if text:
            if not falar_sound_playing:
                speech_engine.say(text)
                speech_engine.runAndWait()
                text_spoken = True 
                falar_sound_playing = True

    else:
        falar_sound_playing = False
        faces = detector(gray)
        for face in faces:
            # x, y = face.left(), face.top()
            # x1, y1 = face.right(), face.bottom()
            # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            # x = landmarks.part(36).x
            # y = landmarks.part(36).y
            # cv2.circle(frame, (x, y), 3, (0, 0, 255), 2)
            
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            if blinking_ratio > 4.8:
                cv2.putText(keyboard, 'PISCAR', (webcam_x + 500, webcam_y + 80), font, 4, (255, 0, 0), thickness=3)
                blinking_frames += 1
                frames -= 1

                # Digitar letra
                if blinking_frames == 5:
                    if active_letter not in ["Apagar", "Falar", "----"]:
                        text += active_letter
                        tecla.play()
                        time.sleep(0.2)
            else:
                blinking_frames = 0

           
            if right_eye_ratio > 8.5:
                cv2.putText(keyboard, 'Apagar', (webcam_x - 300, webcam_y + 80), font, 4, (0, 0, 255), thickness=3)
                
                if text:
                    text = text[:-1]
                    apagar.play()  
                    time.sleep(1)


            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)

            gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

            if gaze_ratio <= 1:
                keyboard_selected = 'Direita'
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
            else:
                keyboard_selected = 'Esquerda'
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected

    # Letras
    if frames == 15:
        letter_index += 1
        frames = 0
    if letter_index == 30:
        letter_index = 0

    for i in range(30):
        if i == letter_index:
            light = True
        else:
            light = False
        letter(i, keys_set_1[i], light)

    text_box_x = 0
    text_box_y = keyboard_height - 100

    text_box_width = keyboard_width
    text_box_height = 100

    # Caixa de texto com a mesma cor das teclas
    text_box = np.zeros((text_box_height, text_box_width, 3), np.uint8)
    text_box[:] = (138, 154, 91)

    # Configurações do texto
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    font_th = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((text_box_width - width_text) / 2)
    text_y = int((text_box_height + height_text) / 2)

    cv2.putText(text_box, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_th)

    # Caixa de texto na janela do teclado
    keyboard[text_box_y:text_box_y + text_box_height, text_box_x:text_box_x + text_box_width] = text_box

    # Redimensionar a imagem da webcam
    webcam = cv2.resize(frame, (webcam_width, webcam_height))

    # Posicionar a imagem da webcam no teclado
    keyboard[webcam_y:webcam_y + webcam_height, webcam_x:webcam_x + webcam_width] = webcam

    #cv2.imshow('Frame', frame)
    cv2.imshow('Teclado Virtual', keyboard)

    # cv2.putText(frame, str(right_side_white), (50, 150), font, 2, (0, 0, 255), 3)
    # threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    # eye = cv2.resize(gray_eye, None, fx=5, fy=5)
    # cv2.imshow('Olho', eye)
    # cv2.imshow('Limite', threshold_eye)
    # #cv2.imshow('Mask', mask)
    # # cv2.imshow('Olho Esquerdo ', left_eye)
    # cv2.imshow(' Limite Esquerdo', left_side_threshold_eye)
    # cv2.imshow(' Limite Direito', right_side_threshold_eye)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
