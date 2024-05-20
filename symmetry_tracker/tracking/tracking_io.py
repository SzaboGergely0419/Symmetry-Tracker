import numpy as np
import cv2
import pandas as pd
import os
import gc
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from symmetry_tracker.general_functionalities.misc_utilities import fig2data

try:
  from IPython.display import display
  from symmetry_tracker.general_functionalities.misc_utilities import progress
except:
  pass

def SaveTracks(AnnotDF, SavePath):
  """
  Saves the AnnotDF dataframe to a json
  """
  if not SavePath.endswith('.json'):
    raise ValueError("SavePath must have a .json extension")
  AnnotDF.to_json(SavePath, orient='records')

def LoadTracks(LoadPath):
  """
  Loads the AnnotDF dataframe from a json
  """
  if not LoadPath.endswith('.json'):
    raise ValueError("LoadPath must have a .json extension")
  AnnotDF = pd.read_json(LoadPath, orient='records')
  return AnnotDF

def WriteTracks(AnnotDF, SavePath):
  """
  Saves the optimal annotation to a txt (standard format)
  Usually should be performed before interpolating missing cell points, as the interpolated values will be saved as regular (unless this behavior is desired)
  """
  with open(SavePath, "w") as OutFile:
    OutFile.write("POS\tF\tCELLNUM\tX\tY\n")
    for Frame in AnnotDF["Frame"].unique():
      for TrackID in AnnotDF.query("Frame == @Frame")["TrackID"].unique():
        Segmentation = coco_mask.decode(AnnotDF.query("Frame == @Frame and TrackID == @TrackID")["SegmentationRLE"].iloc[0]).astype(np.uint8)
        contours, _  = cv2.findContours(Segmentation ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
          for a in contour:
            l=a[0]
            OutFile.write(str(-1)+"\t"+str(Frame+1)+"\t"+str(TrackID)+"\t"+str(l[0])+"\t"+str(l[1])+"\n")
  print("The track is saved in standard txt format to: ")
  print(SavePath)

def DisplayTracks(VideoPath, AnnotDF, DisplayFrameNumber = True, DisplaySegmentation=True, DisplayTrackIDs = True, DisplayPeriodicity=1, Figsize=(5,5)):
  """
  Displays all cell paths as cell IDs on each frame

  - DisplayFrameNumber: Boolean variable marking whether to display the frame number
  - DisplaySegmentation: Boolean variable marking whether to display the segmentations
                          If True, non-interpolated segments are displayed as "lime", interpolated ones are displayed as "deepskyblue"
  - DisplayTrackIDs: Boolean variable marking whether to display the track IDs
  - DisplayPeriodicity: How frequently should the frames be displayed.
                        Useful for long videos.
                        Minimal possible value=1.
  - Figsize: The shape of the figure in standard matplotlib format
  """
  VideoFrames = sorted(os.listdir(VideoPath))
  for Frame in range(len(VideoFrames)):
    if np.mod(Frame,DisplayPeriodicity)==0:

      Img = cv2.cvtColor(cv2.imread(os.path.join(VideoPath, VideoFrames[Frame])), cv2.COLOR_BGR2RGB)
      fig, (ax1) = plt.subplots(1, 1, figsize=Figsize)
      ax1.imshow(Img, cmap=plt.cm.gray, interpolation='nearest')

      if DisplaySegmentation:
        SegmentsSum = np.zeros(np.shape(Img)[0:2])
        for _, Object in AnnotDF.query("Frame == @Frame").iterrows():
          SegmentsSum += coco_mask.decode(Object["SegmentationRLE"])
        ax1.imshow(SegmentsSum, cmap=plt.cm.hot, vmax=2, alpha=.3, interpolation='bilinear')

      if DisplayFrameNumber:
        ax1.text(3, 20, "Frame "+str(Frame+1), color="deepskyblue")

      if DisplayTrackIDs:
        for _, Object in AnnotDF.query("Frame == @Frame").iterrows():
          color="lime"
          if Object["Interpolated"]:
            color="deepskyblue"
          [cx, cy] = Object["Centroid"]
          ax1.text(cx-5, cy+5, Object["TrackID"], color=color)

      plt.show()

def SaveTracksVideo(VideoPath, AnnotDF, OutputVideoPath,
                    Fps = 10, DisplayPeriodicity=1, StartingFrame=None, EndingFrame=None,
                    DisplayFrameNumber=True, DisplaySegmentation=True, ColoredSegmentation=True,
                    DisplayTrackIDs = True, IDFontsize = 10):
  """
  Saves the cell paths in a similar format as DisplayTrack displays them
  Non-interpolated segments are displayed as "lime", interpolated ones are displayed as "deepskyblue"

  - OutputVideoPath: The path to which the video will be saved
  - Fps: The fps of the saved video
  - Starting Frame: The first frame which should be displayed
                  Can be useful if tracking was only performed after a certain frame
  - Ending Frame: The last frame which should be displayed
                  Can be useful if tracking was only performed after a certain frame
  - DisplayPeriodicity: How frequently should the frames be displayed.
                        Useful for long videos.
                        Minimal possible value=1.
  - DisplayFrameNumber: Boolean variable marking whether to display the frame number
  - DisplaySegmentation: Boolean variable marking whether to display the segmentations
  - ColoredSegmentation: Boolean variable marking whether to color the segmentations
  - DisplayTrackIDs: Boolean variable marking whether to display the track IDs
  - IDFontsize: The fontsize of the IDs if they are displayed
  """
  if OutputVideoPath[-4:]!=".mp4":
    raise Exception("Only video paths with mp4 extension are allowed")

  VideoFrames = sorted(os.listdir(VideoPath))
  if StartingFrame is None:
    StartingFrame = 1
  if EndingFrame is None:
    EndingFrame = len(VideoFrames)

  print("Saving Tracks Video")
  try:
    ProgressBar = display(progress(0, EndingFrame-StartingFrame), display_id=True)
  except:
    pass

  out = cv2.VideoWriter(OutputVideoPath, cv2.VideoWriter_fourcc(*'mp4v'), Fps, (700,700), True)
  for Frame in range(StartingFrame-1,EndingFrame-1):
    if np.mod(Frame,DisplayPeriodicity)==0:

      Img = cv2.cvtColor(cv2.imread(os.path.join(VideoPath, VideoFrames[Frame])), cv2.COLOR_BGR2RGB)
      fig, (ax1) = plt.subplots(1, 1, figsize=(7, 7))
      ax1.imshow(Img, cmap=plt.cm.gray, interpolation='nearest')

      if DisplaySegmentation:

        if ColoredSegmentation:
          SegmentsSum = np.zeros_like(Img)
          cmap = cm.nipy_spectral
          for _, Object in AnnotDF.query("Frame == @Frame").iterrows():
              color = cmap((int(Object["TrackID"])*17)%256)
              mask = coco_mask.decode(Object["SegmentationRLE"])*255
              colored_mask = np.array(np.stack([mask*color[0],mask*color[1],mask*color[2]], axis=2),dtype=np.uint8)
              SegmentsSum += colored_mask
          ax1.imshow(SegmentsSum, vmax=256, alpha=.3, interpolation='bilinear')

        else:
          SegmentsSum = np.zeros(np.shape(Img)[0:2])
          for _, Object in AnnotDF.query("Frame == @Frame").iterrows():
            SegmentsSum += coco_mask.decode(Object["SegmentationRLE"])
          ax1.imshow(SegmentsSum, cmap=plt.cm.hot, vmax=2, alpha=.3, interpolation='bilinear')

      if DisplayFrameNumber:
        ax1.text(1, 10, "Frame "+str(Frame+1), color="deepskyblue", fontsize=10)

      if DisplayTrackIDs:
        for _, Object in AnnotDF.query("Frame == @Frame").iterrows():
          color="lime"
          if Object["Interpolated"]:
            color="deepskyblue"
          [cx, cy] = Object["Centroid"]
          ax1.text(cx, cy, Object["TrackID"], color=color, fontsize=IDFontsize, ha='center', va='center')

      Img = cv2.cvtColor(fig2data(fig), cv2.COLOR_BGRA2RGB)
      out.write(Img)
      plt.close(fig)

      try:
        ProgressBar.update(progress(Frame, EndingFrame-StartingFrame))
      except:
        pass

  out.release()
  gc.collect()

  try:
    ProgressBar.update(progress(1, 1))
  except:
    pass