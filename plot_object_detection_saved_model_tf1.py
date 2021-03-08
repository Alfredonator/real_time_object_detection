#!/usr/bin/env python

import os
import cv2
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)
tf.compat.v1.enable_eager_execution()

PATH_TO_MODEL_DIR = "C:\\DATA\\Programming\\models\\workspace\\training_demo\\exported-models\\test_faces"
PATH_TO_LABELS = "C:\\DATA\\Programming\\models\\workspace\\training_demo\\annotations\\faces\\labels.pbtxt"
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "\saved_model"
detect_fn = None
category_index = None


def enable_gpu_support():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("[INFO] GPU support loaded")


def load_model_and_configure_labels():
    print("[INFO] Loading the model started")
    model = tf.saved_model.load_v2(str(PATH_TO_SAVED_MODEL))
    global detect_fn
    detect_fn = model.signatures['serving_default']
    print("[INFO] Model loaded")

    global category_index
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def get_formatted_detections(image):
    return format_detections(get_detections(image))


def format_detections(detections):
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections


def get_detections(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    return detect_fn(input_tensor)


def visualize_detections(image, detections):

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    return image


def main():
    enable_gpu_support()
    load_model_and_configure_labels()

    cap = cv2.VideoCapture(0)
    frame_counter = 0

    while True:

        ret, image_np = cap.read()
        # image_np = cv2.imread('C:\\DATA\\Programming\\models\\workspace\\training_demo\\images\\image_11_39_05.jpg')

        if frame_counter == 0 or frame_counter % 5 == 0:  # capture every 5th frame to lower fps (GPU constraint)

            detections = get_formatted_detections(image_np)
            image_np_with_detections = visualize_detections(image_np, detections)

            frame_counter = 1
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            frame_counter += 1
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()