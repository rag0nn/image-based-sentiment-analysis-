import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, Union
import math
import cv2
import numpy as np
import os

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
  
"""

TODO gap'li face tahmini tek bir yerde olacak yani face direkt gapli olarak tahmin edilce
şunda anootated ögapsiz döndürülüyor
"""
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                        math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(
        image,
        detection_result
    ) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.
    Returns:
        Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                            width, height)
        color, thickness, radius = (0, 255, 0), 2, 2
        cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                        MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image

def detect_face(image:np.ndarray)->Tuple:
    """
    Args:
        image_path
    Returns:
        rgb_annotated_image: Image
        detection_result: Results
    """
    # STEP 2: Create an FaceDetector object.
    base_options = python.BaseOptions(model_asset_path=f'{os.path.dirname(__file__)}/detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # STEP 3: Load the input image.
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )

    # STEP 4: Detect faces in the input image.
    detection_result = detector.detect(mp_image)

    # STEP 5: Process the detection result. In this case, visualize it.
    mp_image_copy = np.copy(mp_image.numpy_view())
    annotated_image = visualize(mp_image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return rgb_annotated_image, detection_result  

def crop_largest_face(image, detection_result, gap_bottom=25, gap_top= 200, gap_left=20, gap_right=20):
    """
    En büyük yüzü kırpar.

    Args:
        image: input RGB numpy array
        detection_result: mediapipe detection result
        gap: yüz crop'una eklenecek boşluk (piksel)

    Returns:
        cropped_face: kırpılmış en büyük yüz (numpy array)
        bbox: (x, y, w, h) bounding box koordinatları
    """
    h_img, w_img, _ = image.shape

    if not detection_result.detections:
        return None, None  # yüz yoksa

    # En büyük yüzü bul
    largest_bbox = None
    max_area = 0
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        area = bbox.width * bbox.height
        if area > max_area:
            max_area = area
            largest_bbox = bbox

    # Bounding box koordinatları
    x = largest_bbox.origin_x
    y = largest_bbox.origin_y
    w = largest_bbox.width
    h = largest_bbox.height

    # Gap ekle ve sınırları kontrol et
    x1 = max(0, x - gap_left)
    y1 = max(0, y - gap_top)
    x2 = min(w_img, x + w + gap_right)
    y2 = min(h_img, y + h + gap_bottom)

    cropped_face = image[y1:y2, x1:x2]

    return cropped_face, (x1, y1, x2 - x1, y2 - y1)