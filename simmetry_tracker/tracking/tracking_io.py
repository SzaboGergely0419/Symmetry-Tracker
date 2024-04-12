import numpy as np
import cv2
import os
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt

from simmetry_tracker.general_functionalities.misc_utilities import fig2data

def WriteTracks(AnnotDF, SavePath):
  """
  Saves the optimal annotation to a txt (standard format)
  Usually should be performed before interpolating missing cell points, as the interpolated values will be saved as regular (unless this behavior is desired)
  """
  #POS argument got removed
  with open(SavePath, "w") as OutFile:
    OutFile.write("F\tCELLNUM\tX\tY\n")
    for Frame in AnnotDF["Frame"].unique():
      for TrackID in AnnotDF.query("Frame == @Frame")["TrackID"].unique():
        Segmentation = coco_mask.decode(AnnotDF.query("TrackID == @TrackID")["SegmentationRLE"].iloc[0]).astype(np.uint8)
        contours, _  = cv2.findContours(Segmentation ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
          for a in contour:
            l=a[0]
            OutFile.write(str(Frame+1)+"\t"+str(TrackID)+"\t"+str(l[0])+"\t"+str(l[1])+"\n")
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

      Img = cv2.imread(os.path.join(VideoPath,VideoFrames[Frame]), cv2.IMREAD_GRAYSCALE)
      fig, (ax1) = plt.subplots(1, 1, figsize=Figsize)
      ax1.imshow(Img, cmap=plt.cm.gray, interpolation='nearest')

      if DisplaySegmentation:
        SegmentsSum = np.zeros_like(Img)
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
                       Fps = 10, DisplayPeriodicity=1, StartingFrame=None, EndingFrame=None, DisplayFrameNumber=True, DisplaySegmentation=True, DisplayTrackIDs = True):
  """
  Saves the cell paths in a similar format as DisplayTrack displays them
  Non-interpolated segments are displayed as "lime", interpolated ones are displayed as "deepskyblue"

  - OutputVideoPath: The path to which the video will be saved
  - DisplayFrameNumber: Boolean variable marking whether to display the frame number
  - DisplaySegmentation: Boolean variable marking whether to display the segmentations
  - DisplayTrackIDs: Boolean variable marking whether to display the track IDs
  - Fps: The fps of the saved video
  - Starting Frame: The first frame which should be displayed
                    Can be useful if tracking was only performed after a certain frame
  - Ending Frame: The last frame which should be displayed
                  Can be useful if tracking was only performed after a certain frame
  - DisplayPeriodicity: How frequently should the frames be displayed.
                        Useful for long videos.
                        Minimal possible value=1.
  """
  if OutputVideoPath[-4:]!=".mp4":
    raise Exception("Only video paths with mp4 extension are allowed")

  VideoFrames = sorted(os.listdir(VideoPath))
  if StartingFrame is None:
    StartingFrame = 1
  if EndingFrame is None:
    EndingFrame = len(VideoFrames)
  out = cv2.VideoWriter(OutputVideoPath, cv2.VideoWriter_fourcc(*'mp4v'), Fps, (1400,1400), True)
  for Frame in range(StartingFrame-1,EndingFrame-1):
    if np.mod(Frame,DisplayPeriodicity)==0:

      Img = cv2.imread(os.path.join(VideoPath,VideoFrames[Frame]), cv2.IMREAD_GRAYSCALE)
      fig, (ax1) = plt.subplots(1, 1, figsize=(14, 14))
      ax1.imshow(Img, cmap=plt.cm.gray, interpolation='nearest')

      if DisplaySegmentation:
        SegmentsSum = np.zeros_like(Img)
        for _, Object in AnnotDF.query("Frame == @Frame").iterrows():
          SegmentsSum += coco_mask.decode(Object["SegmentationRLE"])
        ax1.imshow(SegmentsSum, cmap=plt.cm.hot, vmax=2, alpha=.3, interpolation='bilinear')

      if DisplayFrameNumber:
        ax1.text(1, 10, "Frame "+str(Frame+1), color="deepskyblue", fontsize=20)

      if DisplayTrackIDs:
        for _, Object in AnnotDF.query("Frame == @Frame").iterrows():
          color="lime"
          if Object["Interpolated"]:
            color="deepskyblue"
          [cx, cy] = Object["Centroid"]
          ax1.text(cx-5, cy+5, Object["TrackID"], color=color, fontsize=20)

      Img = cv2.cvtColor(fig2data(fig), cv2.COLOR_BGRA2RGB)
      out.write(Img)
      plt.close()
  out.release()