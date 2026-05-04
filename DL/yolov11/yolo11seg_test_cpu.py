from ultralytics import YOLO

if __name__ == '__main__':
    ### Load your own "best.pt"
    model = YOLO("C:/Users/calmnight/python3/10402AI_course/DL/yolov11/runs/seg/train/weights/best.pt")

    results = model.val(
        data="C:/Users/calmnight/python3/10402AI_course/DL/yolov11/cfg/ai_course_seg.yaml",  # Path to dataset configuration file
        split="test",
        imgsz=640,  # Image size for test
        conf=0.25, # Set confidence based on training results
        iou=0.5,
        plots=True
    )
    print(results.confusion_matrix.to_df())