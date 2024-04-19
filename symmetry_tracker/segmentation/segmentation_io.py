import numpy as np
import cv2
import os
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt

def DisplaySegmentation(VideoPath, Outmasks, DisplayFrameNumber = True, Figsize=(4,4)):
  """
  Displays the segmentation in Outmasks onto the video at VideoPath

  - VideoPath: The video to be displayed in standard .png images format
  - Outmasks: The segmentation results coming from SingleVideoSegmentation()
  - DisplayFrameNumber: Boolean variable marking whether to display the frame number
  - Figsize: The shape of the figure in standard matplotlib format
  """
  VideoFrames = sorted(os.listdir(VideoPath))
  for Frame in range(len(VideoFrames)):
    fig, (ax1) = plt.subplots(1, 1, figsize=Figsize)
    VideoFrame = cv2.cvtColor(cv2.imread(os.path.join(VideoPath, VideoFrames[Frame])), cv2.COLOR_BGR2RGB)
    ax1.imshow(VideoFrame, interpolation='nearest')
    if Frame in Outmasks.keys():
      SegmentsSum = np.zeros(np.shape(VideoFrame)[0:2])
      for segment in Outmasks[Frame]:
        SegmentsSum += coco_mask.decode(segment)
      ax1.imshow(SegmentsSum>0, cmap=plt.cm.hot, vmax=2, alpha=.3, interpolation='bilinear')
      if DisplayFrameNumber:
        ax1.text(3, 20, "Frame "+str(Frame+1), color="deepskyblue")
    plt.show()

def WriteSegmentation(Outmasks, SavePath):
  """
  Saves the segmentation to a txt file

  - Outmasks: The segmentation results coming from SingleVideoSegmentation()
  - SavePath: The path to the file, where the segmentation will be saved
                          If the file does not exists, it will be created
  """
  with open(SavePath, "w") as OutFile:
    OutFile.write("POS\tF\tCELLNUM\tX\tY\n")
    for Frame in Outmasks:
      CellCount=1
      for segment in Outmasks[Frame]:
        contours, hierarchy  = cv2.findContours(np.array(coco_mask.decode(segment), dtype = np.uint8) ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
          for a in contour:
            l=a[0]
            OutFile.write("1"+"\t"+str(Frame+1)+"\t"+"{:04d}".format(Frame+1)+"{:04d}".format(CellCount)+"\t"+str(l[0])+"\t"+str(l[1])+"\n")
          CellCount+=1
  print("Segmentation saved to: "+SavePath)