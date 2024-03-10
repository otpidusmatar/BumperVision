from ultralytics import YOLO

model = YOLO("model/testingbumpermodel.pt")

results = model.track(source="testvideos/robottrimvid.mp4", show=True)

