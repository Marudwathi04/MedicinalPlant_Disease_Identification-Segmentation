if __name__=="__main__":

    from ultralytics import YOLO

    # yolo model creation
    model = YOLO("yolo-weights/yolov8l-obb.pt")
    model.train(data="data.yaml", imgsz=320, batch=6, epochs=50, workers=4)