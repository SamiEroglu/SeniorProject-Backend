from flask import Flask, render_template, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import time
import threading
import base64
import mediapipe as mp

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Function to process image
def process_image(img):
    # Perform image processing here
    # Example:
    processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return processed_img

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

        # Pass the frame to the image processing function
        processed_img = process_image(frame)

        # Display the resulting frames
        cv2.imshow('Hand Detection', frame)
        cv2.imshow('Processed Image', processed_img)

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # time.sleep(0.01)

        # Exit if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
    
