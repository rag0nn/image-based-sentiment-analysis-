from sentinal.sentiment_model import SentimentClassifier,ModelTypes
from test_config import loader, TestTypes
import cv2
import time

def main():
    model_type = ModelTypes.MobileSmall
    model_path=  "/home/enes/Desktop/sentiment-analysis/sentinal/src/sentinal/sentiment_model/mobile_small.pth"
    cs = SentimentClassifier(model_type,model_path)
    
    counter = 0
    test_type = TestTypes.VIDEO
    time_start = time.time()
    
    for label, image in loader(test_type):
        image = cv2.resize(image,(500 ,800))

        h,w,_ = image.shape
        pred, conf = cs.predict(image)
        annotated = cs.visualize(image,pred,conf,"tr")
        print(f"label: {label}, image: {image.shape}, pred_label: {pred}, conf: {conf}")
        
        if pred == label:
            cv2.putText(annotated,str(label),(w-50,h-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(annotated,str(label),(w-50,h-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imshow("annotated", annotated)  
        
        if test_type == TestTypes.VIDEO:
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        else:
            key = cv2.waitKey(0)
            if key == ord("q"):
                break
            
        counter += 1
    
    time_end = time.time()
    if test_type == TestTypes.VIDEO:
        passed = time_end - time_start
        mean_process_time = passed / counter
        mean_fps = 1 / mean_process_time
        print(f"Geçen zaman (saniye): {passed:.1f}", )
        print(f"Ortalama işleme süresi (saniye): {mean_process_time:.3f}")
        print(f"Ortalama fps (saniye): {mean_fps:.3f}" )
        
        

        
        

if __name__ == "__main__":
    main()