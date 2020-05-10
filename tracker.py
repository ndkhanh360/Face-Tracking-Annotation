from __future__ import print_function
import sys
import cv2
from random import randint
import insightface
import argparse
import pickle

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']  

  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
    
  return tracker

def get_bounding_box(frame, trackerType, model=None):
  bboxes = []
  colors = [] 

  if model is not None:
    bboxes, _ = model.detect(frame[:, :, ::-1], threshold=0.5, scale=1.0)
  else:
    while True:
      # draw bounding box
      bbox = cv2.selectROI('MultiTracker', frame)
      bboxes.append(bbox)
      print("Press s to quit selecting boxes and start tracking")
      print("Press any other key to select next object")
      k = cv2.waitKey(0) & 0xFF
      if (k == ord("s")): 
        break
    
  print('Bounding boxes {}'.format(bboxes))

  # Create MultiTracker object
  multiTracker = cv2.MultiTracker_create()

  # Initialize MultiTracker 
  for bbox in bboxes:
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    if len(bbox) == 5:
      bbox[2] -= bbox[0]
      bbox[3] -= bbox[1]
      bbox = tuple(bbox[:-1])
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)

  return [bboxes], colors, multiTracker

def write_annotation(full_annotation, file):
  id = 0
  with open(file, 'w') as f:
    f.write('<?xml version="1.0" encoding="utf-8"?>\n')
    f.write('<annotations>\n')
    lines = []

    for fid_start, annotation in full_annotation:
      n_frames = len(annotation)
      n_obj = len(annotation[0])
        
      for oid in range(n_obj):
        lines.append('<track id="' + str(id) + '" label="Person">\n')
        for fid in range(n_frames):
          try:
            box = annotation[fid][oid]
            lines.append('<box frame="{}" outside="0" occluded="0" keyframe="{}" xtl="{}" ytl="{}" xbr="{}" ybr="{}">\n<attribute name="Emotion">Neutral</attribute>\n</box>\n'.format(\
                fid_start+fid, (1 if not fid else 0), box[0], box[1], box[2]+box[0], box[3]+box[1]))
            lines.append('<box frame="{}" outside="1" occluded="0" keyframe="1" xtl="{}" ytl="{}" xbr="{}" ybr="{}">\n<attribute name="Emotion">Neutral</attribute>\n</box>\n'.format(\
                fid_start+n_frames, box[0], box[1], box[2]+box[0], box[3]+box[1]))
          except:
            pass 
        lines.append('</track>\n')
        id += 1
    
    f.writelines(lines)
    f.write('</annotations>\n')

if __name__=='__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
  ap.add_argument("-t", "--tracker", type=str, default="CSRT",
    help="OpenCV object tracker type")
  ap.add_argument("-m", "--model", type=str, default="True",
    help="whether or not using face detector")
  args = vars(ap.parse_args())

  # Create a list containing all bounding boxes in the videoo
  full_annotation = []

  # Set video to load
  videoPath = args["video"]
  trackerType = args["tracker"]

  # create face model
  model = None
  if args["model"] == "True":
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id = -1, nms=0.4)

  # Create a video capture object to read videos
  cap = cv2.VideoCapture(videoPath)

  # Process video and track objects
  fid = 0
  annotation, colors, multiTracker = None, None, None

  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      break

    fid += 1
    if fid == 1:
      annotation, colors, multiTracker = get_bounding_box(frame, trackerType, model)
    else: 
      key = cv2.waitKey(1) & 0xFF
      if key == ord("q"):  
        break
      elif key == ord("p"):
        full_annotation.append((fid-len(annotation), annotation))
        annotation, colors, multiTracker = get_bounding_box(frame, trackerType, model)

      # get updated location of objects in subsequent frames
      success, boxes = multiTracker.update(frame)
      if success:
        annotation.append(boxes)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
          p1 = (int(newbox[0]), int(newbox[1]))
          p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
          cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
      else:
        annotation.append([])

      # show frame
      cv2.imshow('MultiTracker', frame)

  full_annotation.append((fid-len(annotation), annotation))
  with open('annotation', 'wb') as f:
    pickle.dump(full_annotation, f)
  write_annotation(full_annotation, "output.xml")