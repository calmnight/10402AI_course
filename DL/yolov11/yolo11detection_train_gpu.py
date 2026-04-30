from ultralytics import YOLO

if __name__ == '__main__':
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("yolo11x.pt")

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data="C:/Users/calmnight/python3/10402AI_course/DL/yolov11/cfg/ai_course_detect.yaml",  # Path to dataset configuration file
        workers=4,  # Determined based on your CPU core number
        batch=2,  # Determined based on your GPU RAM
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size for training
        device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

    # Validate the model
    # metrics = model.val()  # no arguments needed, dataset and settings remembered