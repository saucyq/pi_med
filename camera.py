import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from ultralytics import YOLO
import threading
import base64
import json
import requests
from io import BytesIO
from PIL import Image
import random
# pour l affichage une fois la détection
def random_color():
    return ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
# génère les éléments a tester
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
    # nom des éléments
    part_names = list({pin["PinPartName"] for pin in result.get("Pin", [])})
    #génère l image de fin
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


# charge l image pour "voir ou on pointe"
image_path = "body.png"
img = cv2.imread(image_path)
if img is None:
    print("Error: Could not load image.")
    exit()

# modèle squelette
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")
predictor = DefaultPredictor(cfg)

# modèle dispositif
yolo_model = YOLO("best.pt")

class_names = {0: "background", 1: "manche", 2: "dispositif"}

cam = cv2.VideoCapture(0)

while True:
    #reset pour redétecter le dispositif
    if cv2.waitKey(1) & 0xFF == ord('a'):
        send_pick_request.started = False
        print("reset")

    ret, frame = cam.read()
    if not ret:
        break

    outputs = predictor(frame)
    instances = outputs["instances"]
    keypoints = instances.pred_keypoints if instances.has("pred_keypoints") else []

    results = yolo_model.predict(frame, verbose=False)[0]
    masks = results.masks
    classes = results.boxes.cls.int().tolist() if results.boxes is not None else []
    boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []

    output_img = img.copy()

    # affiche les masques yolo
    if masks is not None and len(masks.data) > 0:
        for i, mask in enumerate(masks.data):
            cls_id = classes[i]
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            colored_mask = np.zeros_like(frame)

            if cls_id == 1:
                color = (0, 255, 0)
            elif cls_id == 2:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            colored_mask[mask_np > 0] = color
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

    if len(keypoints) > 0:
        kp = keypoints[0]
        # keypoint de coco
        left_shoulder = kp[5][:2].cpu().numpy()
        right_shoulder = kp[6][:2].cpu().numpy()
        torso_min_x = left_shoulder[0]
        torso_max_x = right_shoulder[0]

        if masks is not None and len(masks.data) > 0:
            for i, mask in enumerate(masks.data):
                cls_id = classes[i]
                if cls_id == 2:
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    ys, xs = np.where(mask_np > 0)
                    if len(xs) == 0:
                        continue
                    median_x_mask = np.median(xs)
                    median_y_mask = np.median(ys)
                    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                    percent = ((median_x_mask - torso_min_x) / (torso_max_x - torso_min_x + 1e-6)) * 100
                    #magic number pour l'image "body.png"
                    static_x_start = 100
                    static_x_end = 377
                    x_on_static_img = int(static_x_start + (percent / 100) * (static_x_end - static_x_start))
                    y_on_static_img = int(100 + (median_y_mask - avg_shoulder_y))
                    #pour pas envoyé plusieurs requetes
                    if not getattr(send_pick_request, "started", False):
                        send_pick_request.started = True
                        threading.Thread(target=send_pick_request, args=(x_on_static_img, y_on_static_img)).start()
                    cv2.circle(output_img, (x_on_static_img, y_on_static_img), 15, (255, 0, 0), -1)
                    cv2.putText(output_img, f"{percent:.1f}%", (x_on_static_img + 10, y_on_static_img),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # dessine les épaules
        cv2.circle(frame, tuple(left_shoulder.astype(int)), 10, (0, 255, 255), -1)
        cv2.circle(frame, tuple(right_shoulder.astype(int)), 10, (0, 255, 255), -1)

    # affiche les détection yolo
    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 255, 0) if cls_id == 1 else (255, 0, 0) if cls_id == 2 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        class_text = class_names.get(cls_id, f"cls{cls_id}")
        cv2.putText(frame, class_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Camera Frame", frame)
    cv2.imshow("Static Image with Class 2 points", output_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
