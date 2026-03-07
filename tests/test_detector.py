"""
Face Recognition ve sentiment modellerini
config.py'deki verilerin path'lerini kullanarak 
ve MODEL_PATH, TEST_TYPE seçeneklerini kullanarak test eder.
"""
import sys
sys.path.append("../")

from .test_config import loader, TestTypes
import cv2
import rerun as rr
import rerun.blueprint as rrb
from sentinal import Sentinal
import numpy as np

TEST_TYPE = TestTypes.IMAGESEQ

def apply(test_type: TestTypes):
    detector = Sentinal()

    # SDK başlatılıyor ve Viewer spawn ediliyor
    rr.init("sentiment_detection", spawn=True)
    
    # 🔵 2D layout blueprint
    rr.send_blueprint(
        rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="image/face_detected"),
                    rrb.Spatial2DView(origin="image/sentiment"),
                    rrb.TimeSeriesView(origin="metrics/classes"),
                    rrb.TimeSeriesView(origin="metrics/confidence"),
                ),
        )
    )

    frame_time = 0.0  # simülasyon zamanı
    dt = 1 / 60.0     # frame süresi

    for real_label, frame_np in loader(test_type):
        # Zamanı set et
        rr.set_time("sim_time", sequence=int(frame_time))

        # Detect
        predictions, annotations = detector.detect(frame_np)
        
        face_annotated_image = annotations[0]
        sentiment_annotates = np.hstack([cv2.resize(im, (400,400)) for im in annotations[1:]])

        # Görselleri logla
        rr.log("image/face_detected", rr.Image(cv2.cvtColor(face_annotated_image,cv2.COLOR_BGR2RGB)))
        rr.log("image/sentiment", rr.Image(cv2.cvtColor(sentiment_annotates, cv2.COLOR_BGR2RGB)))

        # Gerçek sınıf
        rr.log("metrics/classes/real", rr.Scalars(real_label))

        # Tahminler ve confidence log
        for i, pred in enumerate(predictions):
            print(f"Real: {real_label}, Predicted: {pred.pred_lbl}, Conf: {pred.conf:.2f}")
            rr.log(f"metrics/classes/predicted/{i}", rr.Scalars(pred.pred_lbl))
            rr.log(f"metrics/confidence/{i}", rr.Scalars(pred.conf))

        frame_time += dt

if __name__ == "__main__":
    apply(TEST_TYPE)