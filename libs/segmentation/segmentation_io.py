import numpy as np
import cv2
import matplotlib.pyplot as plt

def DisplaySegmentation(Video, Outmasks):
  """
  Displays the segmentation in Outmasks onto the Video
  """
  for Frame in range(np.shape(Video)[0]):
    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 7))
    ax1.imshow(Video[Frame], cmap=plt.cm.gray, interpolation='nearest')
    if Frame in Outmasks.keys():
      SegmentsSum = np.zeros_like(Video[Frame])
      for segment in Outmasks[Frame]:
        SegmentsSum +=segment
      ax1.imshow(SegmentsSum>0, cmap=plt.cm.hot, vmax=2, alpha=.3, interpolation='bilinear')
    plt.show()

def SaveSegmentation(Outmasks, SavePath):
  """
  Saves the segmentation to a txt file

  - SavePath: The path to the file, where the segmentation will be saved 
                          If the file does not exists, it will be created
  """
  with open(SavePath, "w") as OutFile:
    OutFile.write("POS\tF\tCELLNUM\tX\tY\n")
    for Frame in Outmasks:
      CellCount=1
      for segment in Outmasks[Frame]:
        contours, hierarchy  = cv2.findContours( np.uint8(np.array(segment)) ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:    
          for a in contour:
            l=a[0]
            OutFile.write("1"+"\t"+str(Frame+1)+"\t"+str(CellCount)+"\t"+str(l[0])+"\t"+str(l[1])+"\n")
          CellCount+=1
  print("Segmentation saved to: "+SavePath)