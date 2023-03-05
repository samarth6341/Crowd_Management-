import streamlit as st
import supervision as sv
from ultralytics import YOLO
import numpy as np
from PIL import Image 
import cv2

from PIL import Image

model=YOLO("yolov8s.pt")

#detection_output=model.predict(source="C:\Users\samar\Downloads\weights (1)\PIC.jpg",show=False,conf=0.5)
vid_path=f"mallclipmain.mp4"
generator = sv.get_video_frames_generator(vid_path)
iterator = iter(generator)
frame = next(iterator)

results = model(frame, imgsz=1280)[0]
detections = sv.Detections.from_yolov8(results)
detections = detections[detections.class_id == 0]
print(len(detections))

# annotate
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=0.5)
labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

#%matplotlib inline  
sv.show_frame_in_notebook(frame, (16, 16))

# initiate polygon zone
polygon = np.array([
    [820,0],
    [1100,200],
    [1150,300],
    [800,300]
])
video_info = sv.VideoInfo.from_video_path(vid_path)
#print(video_info)
zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

# initiate annotators
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)

# extract video frame
generator = sv.get_video_frames_generator(vid_path)
iterator = iter(generator)
frame = next(iterator)

# detect
results = model(frame, imgsz=1280)[0]
detections = sv.Detections.from_yolov8(results)
detections = detections[detections.class_id == 0]
zone.trigger(detections=detections)

# annotate
box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.25)
labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
frame = zone_annotator.annotate(scene=frame)
Images=Image.fromarray(frame)
#%matplotlib inline  
sv.show_frame_in_notebook(frame,(16, 16))
Images.save("Outputimg.jpg")
