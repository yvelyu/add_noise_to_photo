import os
import csv
from ultralytics import YOLO
from config import UPLOAD_FOLDER

CSV_PATH = "prediction.csv"

MODEL_PATH = "yolo26m.pt" 

def main():
    model = YOLO(MODEL_PATH)
    
    with open(CSV_PATH, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_id", "class_name", "confidence", "bbox"])
        
        for filename in os.listdir(UPLOAD_FOLDER):
            if "histogram" in filename.lower():
                print(f"Пропуск гистограммы: {filename}")
                continue
                
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            
            results = model(img_path)
            
            for result in results:
                if len(result.boxes) == 0:
                    writer.writerow([filename, "", "", "", ""])
                    continue
                    
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = result.names[cls_id]
                    conf = round(float(box.conf[0].item()), 4)
                    bbox = [round(coord, 2) for coord in box.xyxy[0].tolist()]
                    
                    writer.writerow([filename, cls_id, cls_name, conf, bbox])
                    
    print(f"Детекция завершена. Результаты сохранены в {CSV_PATH}")

if __name__ == "__main__":
    main()
