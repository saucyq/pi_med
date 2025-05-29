import cv2
import numpy as np
from ultralytics import YOLO
import threading
import base64
import json
import requests
from io import BytesIO
from PIL import Image
import random

# --- Utilities ---

def random_color():
    return ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

def send_pick_request(x, y):
    pick_url  = "http://lifesciencedb.jp/bp3d/API/pick"
    pick_payload  = {
        "Part": [
            {"PartName":"humerus","PartColor":"0000FF","PartOpacity":0.7},
            {"PartName":"scapula","PartColor":"0000FF","PartOpacity":0.7},
            {"PartName":"clavicle","PartColor":"0000FF","PartOpacity":0.7},
            {"PartName":"supraspinatus","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"brachial plexus","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"axillary nerve","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"musculocutaneous nerve","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"dorsal scapular nerve","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"long thoracic nerve","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"suprascapular nerve","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"nerve to subclavius","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"lateral pectoral nerve","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"medial pectoral nerve","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"upper subscapular nerve","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"lower subscapular nerve","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"thoracodorsal nerve","PartColor":"FF0000","PartOpacity":0.7},
            {"PartName":"pectoralis minor","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"rhomboid major","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"rhomboid minor","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"levator scapulae","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"serratus anterior","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"subscapularis","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"infraspinatus","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"teres minor","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"teres major","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"deltoid","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"biceps brachii","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"coracobrachialis","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"trapezius","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"latissimus dorsi","PartColor":"FFFF00","PartOpacity":0.7},
            {"PartName":"tendon of long head of biceps brachii","PartColor":"D2B48C","PartOpacity":0.7},
            {"PartName":"tendon of long head of triceps brachii","PartColor":"D2B48C","PartOpacity":0.7},
            {"PartName":"axillary fascia","PartColor":"00FF00","PartOpacity":0.7},
            {"PartName":"pectoral fascia","PartColor":"00FF00","PartOpacity":0.7},
            {"PartName":"deltoid fascia","PartColor":"00FF00","PartOpacity":0.7},
            {"PartName":"infraspinous fascia","PartColor":"00FF00","PartOpacity":0.7},
            {"PartName":"supraspinous fascia","PartColor":"00FF00","PartOpacity":0.7},
            {"PartName":"subscapular fascia","PartColor":"00FF00","PartOpacity":0.7},
            {"PartName":"glenohumeral ligaments","PartColor":"A9A9A9","PartOpacity":0.7},
            {"PartName":"coracohumeral ligament","PartColor":"A9A9A9","PartOpacity":0.7},
            {"PartName":"transverse humeral ligament","PartColor":"A9A9A9","PartOpacity":0.7},
            {"PartName":"coracoacromial ligament","PartColor":"A9A9A9","PartOpacity":0.7},
            {"PartName":"acromioclavicular ligament","PartColor":"A9A9A9","PartOpacity":0.7},
            {"PartName":"costoclavicular ligament","PartColor":"A9A9A9","PartOpacity":0.7},
            {"PartName":"glenoid labrum","PartColor":"808080","PartOpacity":0.7}
        ],
        "Window": {"ImageWidth": 500, "ImageHeight": 500},
        "Pick": {"ScreenPosX": x, "ScreenPosY": y}
    }
    print("send pick request")
    print("📤 Pick Request Payload:", json.dumps(pick_payload, separators=(',', ':')))
    pick_response = requests.post(pick_url, data=json.dumps(pick_payload), headers={'Content-Type': 'application/json'})
    if pick_response.status_code != 200:
        print("❌ Pick request failed with status:", pick_response.status_code)
        print("Response text:", pick_response.text)
        exit()

    try:
        result = pick_response.json()
    except json.JSONDecodeError:
        print("❌ Failed to decode JSON from response:")
        print("Raw response:", pick_response.text)
        exit()

    print("✅ Pick API Result:")
    print(json.dumps(result, indent=2))
    result = pick_response.json()
    part_names = list({pin["PinPartName"] for pin in result.get("Pin", [])})

    image_url = 'http://lifesciencedb.jp/bp3d/API/image'
    image_payload = {
        "Part": [{"PartName": "anatomical entity", "PartColor": "F0D2A0", "PartOpacity": 0.1}],
        "Window": {"ImageWidth": 500, "ImageHeight": 500}
    }
    for name in part_names:
        image_payload["Part"].append({
            "PartName": name,
            "PartColor": random_color(),
            "PartOpacity": 0.7
        })

    print("image response ")
    print("URL:", image_url)
    print("📤 Pick Request Payload:", json.dumps(image_payload, separators=(',', ':')))
    image_response = requests.post(
        image_url,
        data=json.dumps(image_payload),
        headers={'Content-Type': 'application/json'}
    )

    print("Image response status code:", image_response.status_code)
    print("Image response headers:", image_response.headers)
    print("Image response content (first 500 chars):", image_response.text[:1000])
    if image_response.ok:
        try:
            response_json = image_response.json()
            data_uri = response_json["data"]

            base64_str = data_uri.split(",")[1]
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            image.show()

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print("Error extracting image from response:", e)
    else:
        print("Image request failed with status:", image_response.status_code)

# --- Async Video Capture Class ---

class VideoCaptureAsync:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.ret, self.frame = self.cap.read()
        self.running = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.running:
            return None
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.read_lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            ret = self.ret
        return ret, frame

    def stop(self):
        self.running = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

    def release(self):
        """Make it behave like cv2.VideoCapture"""
        self.stop()
        self.cap.release()

# --- Load your static image ---
image_path = "body.png"
img = cv2.imread(image_path)
if img is None:
    print("Error: Could not load image.")
    exit()

# --- Load YOLO Models ---

# YOLO keypoint model replacing Detectron2 COCO keypoint model
# Replace "yolo_keypoint_model.pt" with your actual YOLO keypoint model path
yolo_keypoint_model  = YOLO('yolo11n-pose.pt')

# YOLO segmentation/detection model
yolo_segmentation_model = YOLO("best.pt")

class_names = {0: "background", 1: "manche", 2: "dispositif"}

# --- Start async video capture ---
cam = VideoCaptureAsync(0).start()
image_path = "body.png"
img = cv2.imread(image_path)
while True:
    output_img = img.copy()
    ret, frame = cam.read()
    if not ret:
        break

    # Run both models
    kp_results = yolo_keypoint_model(frame, verbose=False)
    seg_results = yolo_segmentation_model(frame, verbose=False)

    # Default values
    keypoints = None
    left_shoulder = None
    right_shoulder = None

    # === Keypoint Detection ===
    for result in kp_results:
        if hasattr(result, "keypoints") and result.keypoints is not None:
            kp_data = result.keypoints
            if kp_data is not None and kp_data.shape[0] > 0:
                kp_numpy = kp_data.data.cpu().numpy()[0]  # (17, 3)
                keypoints = kp_numpy
                left_shoulder = kp_numpy[5][:2]
                right_shoulder = kp_numpy[6][:2]
                # Draw keypoints (shoulders)
                for i in [5, 6]:  # left & right shoulder
                    x, y, conf = kp_numpy[i]
                    if conf > 0.3:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # === Segmentation Detection ===
    for result in seg_results:
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)

        if masks is not None and masks.data is not None and boxes is not None:
            classes = boxes.cls
            for i, mask in enumerate(masks.data):
                cls_id = int(classes[i].item())
                if cls_id == 2:  # Assuming class 2 is your "device"
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    ys, xs = np.where(mask_np > 0)
                    if len(xs) == 0:
                        continue
                    median_x = np.median(xs)
                    median_y = np.median(ys)
                    last_device_location = (median_x, median_y)
                    device_x, device_y = last_device_location
                    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                    y_on_static_img = int(100 + (device_y - avg_shoulder_y))
                    min_x = min(left_shoulder[0], right_shoulder[0])
                    max_x = max(left_shoulder[0], right_shoulder[0])
                    percent = ((device_x - min_x) / (max_x - min_x + 1e-6)) * 100
                    static_x_start = 100
                    static_x_end = 377
                    x_on_static_img = int(static_x_start + (percent / 100) * (static_x_end - static_x_start))
                    cv2.circle(output_img, (x_on_static_img, y_on_static_img), 15, (255, 0, 0), -1)
                    cv2.circle(frame, (int(median_x), int(median_y)), 10, (255, 0, 0), -1)

    # === Handle 'p' key to print positions ===
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        if left_shoulder is not None and right_shoulder is not None:
            print(f"Left shoulder: ({left_shoulder[0]:.1f}, {left_shoulder[1]:.1f})")
            print(f"Right shoulder: ({right_shoulder[0]:.1f}, {right_shoulder[1]:.1f})")
            if last_device_location is not None:
                device_x, device_y = last_device_location
                avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                y_on_static_img = int(100 + (device_y - avg_shoulder_y))
                min_x = min(left_shoulder[0], right_shoulder[0])
                max_x = max(left_shoulder[0], right_shoulder[0])
                percent = ((device_x - min_x) / (max_x - min_x + 1e-6)) * 100
                static_x_start = 100
                static_x_end = 377
                x_on_static_img = int(static_x_start + (percent / 100) * (static_x_end - static_x_start))

                send_pick_request(x_on_static_img,y_on_static_img)
        else:
            print("Shoulders not detected.")

    if key == 27:  # ESC key
        break

    cv2.imshow("Frame", frame)
    cv2.imshow("Static Image with Class 2 points", output_img)

cam.release()
cv2.destroyAllWindows()