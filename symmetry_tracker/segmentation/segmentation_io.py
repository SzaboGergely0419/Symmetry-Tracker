import numpy as np
import cv2
import os
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt

def DisplaySegmentation(VideoPath, AnnotDF, DisplayFrameNumber = True, DisplayPeriodicity = 1, Figsize=(4,4)):
  """
  Displays the segmentation in AnnotDF onto the video at VideoPath

  - VideoPath: The video to be displayed in standard .png images format
  - AnnotDF: The segmentation results coming from SingleVideoSegmentation()
  - DisplayFrameNumber: Boolean variable marking whether to display the frame number
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

      SegmentsSum = np.zeros(np.shape(Img)[0:2])
      for _, Object in AnnotDF.query("Frame == @Frame").iterrows():
        SegmentsSum += coco_mask.decode(Object["SegmentationRLE"])
      ax1.imshow(SegmentsSum, cmap=plt.cm.hot, vmax=2, alpha=.3, interpolation='bilinear')

      if DisplayFrameNumber:
        ax1.text(3, 20, "Frame "+str(Frame+1), color="deepskyblue")

      plt.show()
    
    
def ExportAnnotJSON(AnnotDF, SavePath):
  """
  Saves the AnnotDF dataframe to a json
  All unnecessary columns for the front-end are removed
  """
  if not SavePath.endswith('.json'):
    raise ValueError("SavePath must have a .json extension")
  AnnotDF_export = AnnotDF[['Frame', 'ObjectID', 'SegmentationRLE', 'TrackID', 'Interpolated', 'Class', 'AncestorID']].copy()
  AnnotDF_export.to_json(SavePath, orient='records')