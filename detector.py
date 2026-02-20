import os
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

class OralDetector:
    def __init__(self, model_path='yolo11n.pt'):
        # Load the YOLOv11 model
        self.model = YOLO(model_path)
        # Class names: 0: 'dental caries', 1: 'restoration', 2: 'tooth'
        # (Based on the notebook exploration)
        
    def predict(self, image_path, output_path):
        results = self.model(image_path)[0]
        
        # Load image for custom drawing
        img = cv2.imread(image_path)
        
        tooth_count = 0
        defect_count = 0
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            label = results.names[cls]
            
            # Default color (Green)
            color = (0, 255, 0) 
            
            if label == 'dental caries':
                # RED for defects
                color = (0, 0, 255)
                defect_count += 1
            elif label == 'tooth':
                tooth_count += 1
            elif label == 'restoration':
                # Orange/Yellow for restoration? Let's use Red-ish or separate
                color = (0, 165, 255)
            
            # Draw bounding box
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            
            # Draw label
            text = f"{label} {conf:.2f}"
            cv2.putText(img, text, (int(xyxy[0]), int(xyxy[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Save results
        cv2.imwrite(output_path, img)
        
        return {
            "tooth_count": tooth_count,
            "defect_count": defect_count,
            "label_summary": f"Detected {tooth_count} teeth and {defect_count} defects."
        }

if __name__ == "__main__":
    # Test logic
    detector = OralDetector()
    # If a test image exists, we can run it here.
