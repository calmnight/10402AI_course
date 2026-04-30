from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("C:/Users/nchupmml705/PycharmProjects/YOLO11/pose_eating_detect/20250827_100image_best_test/best.pt")

    results = model.val(data="C:/Users/nchupmml705/PycharmProjects/YOLO11/cfg/chicken-pose.yaml", plots=True)
    print(results.confusion_matrix.to_df())