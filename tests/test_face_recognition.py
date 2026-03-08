from sentinal import FaceDetector
from .test_config import loader, TestTypes
import cv2

def main():
    fd = FaceDetector()
    for label, image in loader(TestTypes.IMAGESEQ,start=1):
        h,w,_ = image.shape
        W = 500
        ratio = h/w
        H = int(ratio * W)
        image = cv2.resize(image,(W,H))
        results = fd.detect_face(image)
        results = fd.add_margin(image,results)
        annotated_image = fd.visualize(image, results)
        faces = fd.crop_faces(image, results)

        cv2.imshow("annotated", annotated_image)
        for i,im in enumerate(faces):
            cv2.imshow(f"{i}", im)
            
        cv2.waitKey(0)
        

if __name__ == "__main__":
    main()