import cv2
from ultralytics import YOLO
from flask import Flask, Response, render_template

app = Flask(__name__)

# Load the YOLO model
model = YOLO('/Users/alaaalseeni/Desktop/best.pt')  # Ensure the path to your trained model is correct

# Load and resize the small image
image_path = '/Users/alaaalseeni/Desktop/week4project/im.jpeg'  # Path to the small image to overlay
small_image = cv2.imread(image_path)
resize_scale = 0.5  # Resize scale factor
small_image = cv2.resize(small_image, (0, 0), fx=resize_scale, fy=resize_scale)
small_image_height, small_image_width = small_image.shape[:2]

# Video source
video_path = '/Users/alaaalseeni/Desktop/week4project/vid.mp4'

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Extract results
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs

        # Get class names (you need to load the class names or labels according to your model)
        class_names = model.names  # This should be a list of class names

        # Count detected vehicles
        vehicle_count = 0

        # Annotate the frame
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            if class_names[int(class_id)] == "car":  # Change "car" to match your class name for vehicles
                label = f'Vehicle {score:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                vehicle_count += 1

        # Display the vehicle count on the frame
        cv2.putText(frame, f'Vehicle Count: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Get the current timestamp in seconds
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Overlay the small image at 11 seconds and hide it after 4 seconds
        if 11 <= current_time <= 15:
            y_offset = 10
            x_offset = frame_width - small_image_width - 10
            frame[y_offset:y_offset + small_image_height, x_offset:x_offset + small_image_width] = small_image

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
