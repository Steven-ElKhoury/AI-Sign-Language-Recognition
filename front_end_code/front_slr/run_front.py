
from flask import Flask, jsonify
from flask_cors import CORS
from flask import render_template,Response

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from flask_socketio import SocketIO

mp_holistic = mp.solutions.holistic # The holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  
    results = model.process(image)    
    image.flags.writeable = True             
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

#open cv uses BGR instead of RGB
# mediapipe uses RGB

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, left_hand, right_hand])

# DATA_PATH = os.path.join('MP_Data')
actions = np.array([str(i) for i in range(1, 66)])
# no_sequences = 10 
# video_num_length = 5
# start_folder = 0
# sequences = np.load('sequences.npy')
# labels = np.load('labels.npy')

# from keras.models import Sequential
# from keras.layers import LSTM, Dropout, Dense
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold

# model = Sequential()
# model.add(LSTM(128, return_sequences=True, input_shape=(30, 1662))) 
# model.add(LSTM(128, return_sequences=True))
# model.add(LSTM(128))
# model.add(Dense(64, activation='softmax'))  

# # Compile the model
# optimizer = Adam(learning_rate=0.001)
# early_stopping = EarlyStopping(monitor='val_loss', patience=20)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# X = sequences
# y= to_categorical(labels).astype(int)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)    
from keras.models import load_model

model = load_model('actions\\actions_double_gru.h5')

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

mp_holistic = mp.solutions.holistic

label_map_int = {
    1: "Opaque",
    2: "Red",
    3: "Green",
    4: "Yellow",
    5: "Bright",
    6: "Light-blue",
    7: "Colors",
    8: "Pink",
    9: "Women",
    10: "Enemy",
    11: "Son",
    12: "Man",
    13: "Away",
    14: "Drawer",
    15: "Born",
    16: "Learn",
    17: "Call",
    18: "Skimmer",
    19: "Bitter",
    20: "Sweet milk",
    21: "Milk",
    22: "Water",
    23: "Food",
    24: "Argentina",
    25: "Uruguay",
    26: "Country",
    27: "Last name",
    28: "Where",
    29: "Mock",
    30: "Birthday",
    31: "Breakfast",
    32: "Photo",
    33: "Hungry",
    34: "Map",
    35: "Coin",
    36: "Music",
    37: "Ship",
    38: "None",
    39: "Name",
    40: "Patience",
    41: "Perfume",
    42: "Deaf",
    43: "Trap",
    44: "Rice",
    45: "Barbecue",
    46: "Candy",
    47: "Chewing-gum",
    48: "Spaghetti",
    49: "Yogurt",
    50: "Accept",
    51: "Thanks",
    52: "Shut down",
    53: "Appear",
    54: "To land",
    55: "Catch",
    56: "Help",
    57: "Dance",
    58: "Bathe",
    59: "Buy",
    60: "Copy",
    61: "Run",
    62: "Realize",
    63: "Give",
    64: "Find",
    65: 'My Sign'
}





sequence = [] # collect frames to make our predictions
sentence = [] # store the sentence that we are going to predict
threshold = 0.9 # threshold for prediction (only render results above this threshold)
predictions = [] # This will prevent detecting signs by mistake as we transition between signs
capturing_frames = False # Flag to indicate whether frames should be captured

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
def generate_frames():
    frame_count = 0 # Counter to keep track of frames captured

    last_frame_time = time.time()
    answer =str(0)
    camera=cv2.VideoCapture(0)
    while True:
        current_time = time.time()
        elapsed_time = current_time - last_frame_time

        ## read the camera frame
        if elapsed_time > 1 / 10:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Draw landmarks
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                keypoints = extract_keypoints(results)



                if frame_count == 0 or frame_count == 29:
                    capturing_frames = True
                    sequence = []  # Reset sequence
                    frame_count = 0
                    cv2.waitKey(2000)

                    time_init = time.time()

                if capturing_frames:
                    sequence.append(keypoints)
                    frame_count += 1
                    #print(frame_count)
                    if frame_count == 29: # Capture frames for 30 frames
                        capturing_frames = False
                        
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        predictions.append(np.argmax(res))
                        print(actions[np.argmax(res)])
                        answer = str(label_map_int[int(actions[np.argmax(res)])])
                        #answer = str(time_init -time.time()

                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, answer, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                ret,buffer=cv2.imencode('.jpg',image)
                frame=buffer.tobytes()

            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            last_frame_time = current_time



app = Flask(__name__, template_folder='./react_front/front_slr/public/')

CORS(app)  # Enable CORS for all routes

# Dummy data representing model information
model_data = [
    {"model": "CNN followed by LSTM", "model_accuracy": 92.3},
    {"model": "Double GRU followed by LSTM", "model_accuracy": 85.38},
    {"model": "LSTM followed by GRU", "model_accuracy": 87.07},
    {"model": "KNN", "model_accuracy": 54.46},
    {"model": "GRU followed by LSTM", "model_accuracy": 84.15},
    {"model": "BLSTM", "model_accuracy": 87.69},
    {"model": "Double GRU", "model_accuracy": 94},
    {"model": "Triple LSTM", "model_accuracy": 90},
    {"model": "My Sign Language Model", "model_accuracy": 0.95}
]


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/api/data')
def get_model_data():
    return jsonify(model_data)

if __name__ == '__main__':
    app.run(debug=True)
    is_streaming = False

