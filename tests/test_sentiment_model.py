from sentinal import SentimentClassifier,ModelTypes
from .test_config import loader, TestTypes
import cv2

def main():
    model_type = ModelTypes.Resnet101
    model_path=  "/home/enes/Desktop/sentiment-analysis/sentinal/src/sentiment_model/best.pth"
    cs = SentimentClassifier(model_type,model_path)
    for label, image in loader(TestTypes.IMAGESEQ):
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
        
        cv2.waitKey(0)
        break
        
        

if __name__ == "__main__":
    main()