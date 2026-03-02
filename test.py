"""
Face Recgonition ve sentiment modellerini 
config.py'deki verilerin path'lerini kullanarak 
ve aşağıda tanımlanan MODEL_PATH, TEST_TYPE 
seçeneklerini kullanarak test eder ve ekranda gösterir.
"""
from project_test.config import TestTypes, loader
import cv2
import rerun as rr
import rerun.blueprint as rrb
from detector import SentimentDetector

TEST_TYPE = TestTypes.VIDEO

def apply(test_type:TestTypes):
    detector = SentimentDetector()
    
    rr.init("sentiment_detection", spawn=True)
    
    # 🔵 2D layout blueprint
    rr.send_blueprint(
        rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="image/face_detected"),
                    rrb.Spatial2DView(origin="image/face_cropped"),
                    rrb.TimeSeriesView(origin="metrics/classes"),
                    rrb.TimeSeriesView(origin="metrics/confidence"),
                ),
        )
    )

    frame_time = 0.0 # rerun time params
    dt = 1 / 60.0   # rerun time params
    for real_label, frame_np in loader(test_type):
        rr.set_time("sim_time", duration=frame_time)
        
        predicted_class, confidence, cropped_face_image, face_xywh, face_annotated_image = detector.apply(frame_np)
        
        print("Real-> ", real_label, "  Predicted-> ", predicted_class)
        rr.log("image/face_detected", rr.Image(face_annotated_image))
        rr.log("image/face_cropped", rr.Image(cv2.cvtColor(cropped_face_image,cv2.COLOR_BGR2RGB)))
        
        rr.log("metrics/classes/real", rr.Scalars(real_label))
        rr.log("metrics/classes/predicted", rr.Scalars(predicted_class))
        rr.log("metrics/confidence", rr.Scalars(confidence))
        
        # Bir sonraki frame zamanı
        frame_time += dt
        
if __name__ == "__main__":
    apply(TEST_TYPE)