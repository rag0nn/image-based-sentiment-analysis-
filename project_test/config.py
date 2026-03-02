from enum import Enum
import cv2
from typing import Any, Generator, Tuple

data_img_dict = {
    0: ["/home/enes/Desktop/sentiment-analysis/project_test/1772293663003.jpg",
        ],
    1: ["/home/enes/Desktop/sentiment-analysis/project_test/1772293662993.jpg",
        ],
    2: ["/home/enes/Desktop/sentiment-analysis/project_test/1772293662983.jpg",
        ],
    3: ["/home/enes/Desktop/sentiment-analysis/project_test/1772293662973.jpg",
        ],
    4: ["/home/enes/Desktop/sentiment-analysis/project_test/1772293662960.jpg",
        ],
    5: ["/home/enes/Desktop/sentiment-analysis/project_test/1772293662950.jpg",
        ],
    6: ["/home/enes/Desktop/sentiment-analysis/project_test/1772293662940.jpg",
        ],
    7: ["/home/enes/Desktop/sentiment-analysis/project_test/1772293662927.jpg",
        ],
}
data_vid_list = [
    "/home/enes/Desktop/sentiment-analysis/project_test/VID_20260228_183441.mp4",
    # "",
]



class TestTypes(Enum):
    VIDEO = 0
    IMAGESEQ = 1
    
def loader(test_type:TestTypes)-> Generator[Tuple[Any, cv2.Mat], None, None]:
    if test_type == TestTypes.VIDEO:
        for vid_path in data_vid_list:
            cap = cv2.VideoCapture(vid_path)

            if not cap.isOpened():
                print("Video açılamadı.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                yield None, frame

    elif test_type == TestTypes.IMAGESEQ:
        for k,v in data_img_dict.items():
            for img_path in v:
                image = cv2.imread(img_path)
                yield k, image
        