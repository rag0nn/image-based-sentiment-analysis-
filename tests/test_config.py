from enum import Enum
import cv2
from typing import Any, Generator, Tuple
import math

# Test Data Paths
data_img_dict = {
    0: ["/home/enes/Desktop/sentiment-analysis/other/data/test/1772293663003.jpg",
        ],
    1: ["/home/enes/Desktop/sentiment-analysis/other/data/test/1772293662993.jpg",
        "/home/enes/Desktop/sentiment-analysis/other/data/test/1772932573832.jpg"
        ],
    2: ["/home/enes/Desktop/sentiment-analysis/other/data/test/1772293662983.jpg",
        ],
    3: ["/home/enes/Desktop/sentiment-analysis/other/data/test/1772293662973.jpg",
        ],
    4: ["/home/enes/Desktop/sentiment-analysis/other/data/test/1772293662960.jpg",
        ],
    5: ["/home/enes/Desktop/sentiment-analysis/other/data/test/1772293662950.jpg",
        ],
    6: ["/home/enes/Desktop/sentiment-analysis/other/data/test/1772293662940.jpg",
        ],
    7: ["/home/enes/Desktop/sentiment-analysis/other/data/test/1772293662927.jpg",
        ],
}
data_vid_list = [
    "/home/enes/Desktop/sentiment-analysis/other/data/test/VID_20260228_183441.mp4"
    # "",
]


# Data Loader
class TestTypes(Enum):
    VIDEO = 0
    IMAGESEQ = 1
    
def loader(test_type:TestTypes, start=0, end=math.inf)-> Generator[Tuple[Any, cv2.Mat], None, None]:
    if start > end:
        raise KeyError("Start can't be bigger than end")
    
    if test_type == TestTypes.VIDEO:

        for vid_path in data_vid_list:
            cap = cv2.VideoCapture(vid_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            if not cap.isOpened():
                print("Video açılamadı.")
                return

            counter = start
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if counter == end:
                    break
                counter +=1
                yield None, frame

    elif test_type == TestTypes.IMAGESEQ:
        for k,v in data_img_dict.items():
            counter = 0
            for img_path in v:
                try:
                    image = cv2.imread(img_path)
                except Exception as e:
                    raise Exception(f"{img_path} bulunamadı")
                if start > counter:
                    counter += 1
                    continue
                if end == counter:
                    break
                
                counter += 1
                yield k, image
        