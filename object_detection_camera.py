#!/usr/bin/env python
# coding: utf-8

import os
import tarfile
import urllib.request
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes_detected = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes_detected)
    detections_detected = detection_model.postprocess(prediction_dict, shapes_detected)

    return detections_detected, prediction_dict, tf.reshape(shapes_detected, [-1])


DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for directory in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.mkdir(directory)

"""
# Download and extract model
"""

MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
if not os.path.exists(PATH_TO_CKPT):
    print('Downloading model. This may take a while... ', end='')
    urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
    tar_file = tarfile.open(PATH_TO_MODEL_TAR)
    tar_file.extractall(MODELS_DIR)
    tar_file.close()
    os.remove(PATH_TO_MODEL_TAR)
    print('Done')

# Download labels file
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
if not os.path.exists(PATH_TO_LABELS):
    print('Downloading label file... ', end='')
    urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
    print('Done')


"""
# Load the model
"""

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()


# Load label map data (for plotting)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

"""
# Capture video frame and do the detection
"""

cap = cv2.VideoCapture(0)
frame_counter = 0

while True:

    # Read frame from camera
    ret, image_np = cap.read()

    if frame_counter == 0 or frame_counter % 5 == 0:

        image_np_expanded = np.expand_dims(image_np, axis=0)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        print(f"before detection, counter: {frame_counter}")
        detections, predictions_dict, shapes = detect_fn(input_tensor)
        print("after detection")

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        frame_counter = 1

        # Display output
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        frame_counter += 1
        continue


    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # image_np_expanded = np.expand_dims(image_np, axis=0)
    #
    # input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    # print(f"before detection, counter: {counter}")
    # detections, predictions_dict, shapes = detect_fn(input_tensor)
    # print("after detection")
    #
    # label_id_offset = 1
    # image_np_with_detections = image_np.copy()
    #
    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #       image_np_with_detections,
    #       detections['detection_boxes'][0].numpy(),
    #       (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
    #       detections['detection_scores'][0].numpy(),
    #       category_index,
    #       use_normalized_coordinates=True,
    #       max_boxes_to_draw=200,
    #       min_score_thresh=.30,
    #       agnostic_mode=False)
    #
    # counter = 0
    #
    # # Display output
    # cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    #
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break


cap.release()
cv2.destroyAllWindows()
