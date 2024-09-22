import openvino as ov
import cv2
import numpy as np
import argparse
core = ov.Core()
parser = argparse.ArgumentParser(description='demo for taguchi_face_recognition_resnet_v1_openvino_fp16_optimized.')
parser.add_argument('--faces_to_be_compared', "-faces", type=str, default="./asset/1_faces_to_be_compared.jpg", help="Usage: Set 1 image file path.")
parser.add_argument('--face_to_compare', "-face", type=str, default="./asset/1_face_to_compare.jpg", help="Usage: Set 1 image file path.")
parser.add_argument('--tolerance', type=float, default=0.4)
parser.add_argument('--device', type=float, default=0.4)

args, unknown = parser.parse_known_args()

model_path = "../taguchi_face_recognition_resnet_v1_openvino_fp16_optimized.xml"
device_name = "CPU" #"GPU", "NPU" may possible depends on your PC.

if device_name not in core.available_devices: # ['CPU', 'GPU', 'NPU']
    device_name = core.available_devices[0]

# Face recognizer
# For this model, you need to add the batch size 1 to shepe it from [?, 150, 150, 3] to [1, 150, 150, 3]
face_recognition_model = core.read_model(model_path)
face_recognition_model.reshape([1, 150, 150, 3])
face_recognizer = core.compile_model(model=face_recognition_model, device_name=device_name)

# Face detector: use Haar Cascades from opencv
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def convert_color_space(src):
    if len(src.shape) == 3 and src.shape[2] == 4:
        # non-animation with alpha channel
        return cv2.cvtColor(src, cv2.COLOR_BGR2RGBA)
    elif len(src.shape) == 4:
        # gif/png animation with alpha channel
        for i in range(src.shape[0]):
            src[i] =cv2.cvtColor(src[i], cv2.COLOR_BGR2RGBA)
        return src
    else:
        # nothing needed for image without alpha channel
        return src
    
def face_detection(image):
    face_locations = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 顔を検出
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    # 検出された顔の位置を記録
    for (left, top, width, height) in faces:
        face_locations.append([left, top, width, height])
    return face_locations

def face_encoding(image, face_locations):
    encodings = []
    for face in face_locations:
        left, top, width, height = face 
        face_roi = image[top:top+height, left:left+width]
        preprocessed_image = cv2.resize(face_roi, (150,150)) # 入力画像の前処理
        preprocessed_image = preprocessed_image / 255.0  # 正規化
        input_data = np.expand_dims(preprocessed_image, axis=0).astype(np.float32)
        face_feature = face_recognizer(input_data)[face_recognizer.output(0)][0]
        encodings.append(face_feature)
    return encodings

def face_recognition(faces_to_be_compared, face_to_compare, tolerance=args.tolerance):
    """
    return most similar face as True.
    Similarity is calcurated by L2.
    """
    if len(faces_to_be_compared) == 0:
        print(f"No face is detected in file:{args.faces_to_be_compared}.")
        return []
    if len(face_to_compare) != 1:
        print(f"Face to compare must be 1. It contains :{len(face_to_compare)}. 1st face is used to detect.")
    face_to_compare = face_to_compare[0]
    distances = [np.linalg.norm(face -face_to_compare) for face in faces_to_be_compared]
    min_distance = min(min(distances), tolerance) 
    return [(x, x <= min_distance) for x in distances] # list of (distance, True or False)

if __name__ == "__main__":
    faces_to_be_compared = cv2.imread(args.faces_to_be_compared)

    # 誤検知削減のため画像の幅を1240以下にリサイズ
    height, width = faces_to_be_compared.shape[:2]
    if width > 1240:
        new_width = 1240
        new_height = int((new_width / width) * height)
        faces_to_be_compared = cv2.resize(faces_to_be_compared, (new_width, new_height))
 
    faces_to_be_compared = convert_color_space(faces_to_be_compared)
    location_faces_to_be_compared = face_detection(faces_to_be_compared)
    encoding_faces_to_be_compared = face_encoding(faces_to_be_compared, location_faces_to_be_compared)

    face_to_compare = cv2.imread(args.face_to_compare)
    # 誤検知削減のため画像の幅を200以下にリサイズ   
    height, width = face_to_compare.shape[:2]
    if width > 200:
        new_width = 200
        new_height = int((new_width / width) * height)
        face_to_compare = cv2.resize(face_to_compare, (new_width, new_height))

    face_to_compare = convert_color_space(face_to_compare)
    location_face_to_compare = face_detection(face_to_compare)
    encoding_face_to_compare = face_encoding(face_to_compare, location_face_to_compare)

    results = face_recognition(encoding_faces_to_be_compared, encoding_face_to_compare, args.tolerance)
    print("faces_to_be_compared:", args.faces_to_be_compared, "\nface_to_compare:     ", args.face_to_compare )
    for result in results:
        print(f"simirality:{result[0]:.3f}, result: {result[1]}")
    
    # 認識された顔にバウンディングボックスと信頼度を描画
    for (i, (x, y, w, h)) in enumerate(location_faces_to_be_compared):
        distance, result = results[i]
        text = f'{distance:.3f}'
        color = (255, 0, 0) if result else (0, 0, 255)
        cv2.rectangle(faces_to_be_compared, (x, y), (x+w, y+h), color, 2)
        cv2.putText(faces_to_be_compared, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 検出された顔にバウンディングボックスを表示
    for (i, (x, y, w, h)) in enumerate(location_face_to_compare):
        color = (255, 0, 0)
        cv2.rectangle(face_to_compare, (x, y), (x+w, y+h), color, 2)

    # 結果を表示
    cv2.imshow('Detected Faces', faces_to_be_compared)
    cv2.imshow('Faces to compare', face_to_compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

