from ultralytics import YOLO

model = YOLO("main/model/testingbumpermodel.pt")

results = model.track(source="main/testvideos/robottrimvid.mp4", show=True)

