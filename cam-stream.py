from flask import Flask, render_template, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import time
import threading
import base64
import mediapipe as mp
import tensorflow as tf
import numpy as np

from tensorflow import keras

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')

# Load the trained model
model = keras.models.load_model('the_best_model.h5')

# Function to process image
# def process_image(img):
#     # Perform image processing here
#     # Example:
#     processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     return processed_img

def process_image(img):
    # Perform image processing here
    # Example:
    processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize the processed image to match the input size of the model
    processed_img = cv2.resize(processed_img, (224, 224))

    # Convert the image to RGB color space
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)

    # Add a batch dimension to the processed image
    processed_img = np.expand_dims(processed_img, axis=0)

    # Normalize the processed image
    processed_img = processed_img / 255.0

    return processed_img


# Initialize the Mediapipe Holistic solution
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def generate_frames():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Loop through each frame from the webcam
    while True:
        ret, frame = cap.read()

        # Convert the image to RGB color space
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect the hand landmarks using Holistic
        results = holistic.process(image)
        
        if results.right_hand_landmarks or results.left_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # # Pass the frame to the image processing function
        # processed_img = process_image(frame)

        # # Normalize the processed image and resize it to match the input size of the model
        # # processed_img = cv2.resize(processed_img, (64, 64)) / 255.0
        # processed_img = cv2.resize(processed_img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

        # # Reshape the processed image to match the input shape of the model
        # processed_img = np.reshape(processed_img, (1, 64, 64, 1))

        # # Predict the label for the processed image
        # label = model.predict(processed_img)

        # # Display the predicted label on the frame
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(frame, str(label), (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # # Display the resulting frames
        # cv2.imshow('Hand Detection', frame)
        # # cv2.imshow('Processed Image', processed_img)

        # ret, buffer = cv2.imencode('.jpg', frame)
        # frame = buffer.tobytes()

        time.sleep(1)

        # Pass the frame to the image processing function
        processed_img = process_image(frame)

        # Predict the label for the processed image
        predictions = model.predict(processed_img)

        print("Predictions : ", predictions)
        print("Class id : ", np.argmax(predictions))
        print(np.sort(predictions))

        labels = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

        image_class = labels[np.argmax(predictions)]
        print("Image class :", image_class)

        # Display the predicted label on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(image_class), (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frames
        cv2.imshow('Hand Detection', frame)

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        # yield the labeled frame
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Exit if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app, cors_allowed_origins='*')

# # Initialize Mediapipe Holistic
# mp_holistic = mp.solutions.holistic
# holistic = mp_holistic.Holistic()

# # Function to process image
# def process_image(img):
#     # Perform image processing here
#     # Example:
#     processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     return processed_img


# def generate_frames():
#     # Open the webcam
#     cap = cv2.VideoCapture(0)

#     # Loop through each frame from the webcam
#     while True:
#         ret, frame = cap.read()

#         # Convert the image to RGB color space
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Detect the hand landmarks using Holistic
#         results = holistic.process(image)
        
#         if results.right_hand_landmarks or results.left_hand_landmarks:
#             mp_drawing = mp.solutions.drawing_utils
#             mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#             mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

#         # Pass the frame to the image processing function
#         processed_img = process_image(frame)

#         # Display the resulting frames
#         cv2.imshow('Hand Detection', frame)
#         cv2.imshow('Processed Image', processed_img)

#         ret,buffer=cv2.imencode('.jpg',frame)
#         frame=buffer.tobytes()

#         # frame'i modele gönderip
#         # modelden gelen işaretli sonuçları
#         # da yield ile atmak lazım

#         yield(b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
#         # time.sleep(0.01)

#         # Exit if the user presses the 'q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break



# socket on data
@socketio.on('data')
def data(data):
    print('Received data:', data)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    
    # app.run(debug=True)
    print('Server started on 5000')
    t = threading.Thread(socketio.run(app, host='0.0.0.0', port=5000))
    t.start()

    

    # app.run(debug=True)
    # socketio.start_background_task(capture_frames)
    
