from flask import Flask, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import time
import base64

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')


# works for 100 frames
def capture_frames():
    start_time = time.time()
    capture = cv2.VideoCapture(0)
    frame_count = 0
    while True:
        success, frame = capture.read()
        if success:
            frame_name = "framevideo.jpg"
            # extracting frames from video
            cv2.imwrite(frame_name, frame)

            with open(frame_name, 'rb') as f:
                frame_data = f.read()
                frame_b64 = base64.b64encode(frame_data).decode('utf-8')
                yield (frame_b64)

            frame_count += 1

            # emit the frame to the SocketIO server
            socketio.emit('frame', frame_b64)

        # Stop sending frames after 10 seconds or 100 frames
        if time.time() - start_time > 10 or frame_count >= 100:
            break



@app.route('/cam', methods=['GET'])
def cam():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5003)
