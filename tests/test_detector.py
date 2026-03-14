"""
Face Recognition ve sentiment modellerini
config.py'deki verilerin path'lerini kullanarak 
ve MODEL_PATH, TEST_TYPE seçeneklerini kullanarak test eder.
"""

from test_config import loader, TestTypes
from sentinal.sentiment_model.structs import EMOTION_DICT, EMOTION_DICT_TR
import cv2
import rerun as rr
import rerun.blueprint as rrb
from sentinal.detector import Sentinal, Models
import logging

TEST_TYPE = TestTypes.VIDEO
WINDOW_H = 700

# LOGGERS
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

def apply_with_rerun(test_type: TestTypes):
    detector = Sentinal(Models.MobileSmall)
    # SDK başlatılıyor ve Viewer spawn ediliyor
    rr.init("sentiment_detection", spawn=True)
    
    # 🔵 2D layout blueprint
    rr.send_blueprint(
        rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="image/annotated"),
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
        predictions = detector.detect(frame_np)
        annotated = detector.visualize(frame_np.copy(),predictions,EMOTION_DICT_TR)
        
        # Görselleri logla
        rr.log("image/annotated", rr.Image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)))

        # Gerçek sınıf
        rr.log("metrics/classes/real", rr.Scalars(real_label))

        # Tahminler ve confidence log
        for i, pred in enumerate(predictions):
            print(f"Real: {real_label}, Predicted: {pred.pred_lbl}, Conf: {pred.conf:.2f}")
            rr.log(f"metrics/classes/predicted/{i}", rr.Scalars(pred.pred_lbl))
            rr.log(f"metrics/confidence/{i}", rr.Scalars(pred.conf))

        frame_time += dt


def apply_with_cv2(test_type):
    detector = Sentinal(Models.MobileSmall)
    frame_time = 0
    for real_label, frame_np in loader(test_type):
        # Detect
        predictions = detector.detect(frame_np)
        annotated = detector.visualize(frame_np.copy(),predictions,EMOTION_DICT_TR)
        # Tahminler ve confidence log
        for i, pred in enumerate(predictions):
            print(f"Real: {real_label}, Predicted: {pred.pred_lbl}, Conf: {pred.conf:.2f}")        
        h,w,_ = frame_np.shape
        ratio = w/h
        new_w = int(ratio * WINDOW_H)
        annotated = cv2.resize(annotated, (new_w,WINDOW_H))
        cv2.imshow("Annotated", annotated)            
        if test_type == TestTypes.IMAGESEQ:
            key = cv2.waitKey(0)
            if key == ord("q"):
                break
            else:
                pass
        else:
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        frame_time += 1


if __name__ == "__main__":
    apply_with_rerun(TEST_TYPE)