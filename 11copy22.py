import os
import torch
import streamlit as st
import supervision as sv
from ultralytics import YOLO
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from PIL import Image
st.set_page_config(layout='wide')
imagess=Image.open('logo.png')

with st.sidebar: 
    st.image(imagess,caption=None)
    st.title('CrowdView')

model=YOLO("yolov8s.pt")
ppl=[]
timestamps=[]

#detection_output=model.predict(source="C:\Users\samar\Downloads\weights (1)\PIC.jpg",show=False,conf=0.5)
vid_path=f".mp4"
video_file = open('idkk.mp4','rb')
video_bytes = video_file.read()
st.title("MALL VIEW")
st.video(video_bytes)

generator = sv.get_video_frames_generator(vid_path)
iterator = iter(generator)
frame = next(iterator)
results = model(frame, imgsz=1280)[0]
detections = sv.Detections.from_yolov8(results)
detections = detections[detections.class_id == 0]
chart_data = pd.DataFrame(np.array(timestamps),np.array(ppl))

st.bar_chart(chart_data)



def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_yolov8(results)
    detections = detections[detections.class_id == 0]
    no=len(detections)
    ppl.append(no)
    #zone.trigger(detections=detections)


    # annotate
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        #frame = zone_annotator.annotate(scene=frame)
    current_time = datetime.datetime.now()
    timestamps.append(current_time)
    

    return frame
sv.process_video(source_path=vid_path, target_path=f"new44.mp4", callback=process_frame)
print(ppl)
#print(time)
timestamps.reverse()

plt.plot(timestamps,ppl)
plt.show()


vid_path="new44.mp4"
video_file2 = open('new44.mp4','rb')
video_bytes2= video_file2.read()
chart_data = pd.DataFrame(np.array(timestamps),np.array(ppl))

st.bar_chart(chart_data)

st.video(video_bytes2)