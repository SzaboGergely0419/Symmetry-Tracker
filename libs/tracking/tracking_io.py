import numpy as np
import cv2
import lzma
import pickle
import matplotlib.pyplot as plt

from tracking.tracker_utilities import AnnotToPaths, CalculateAnnotFramewiseSum
from general_functionalities.misc_utilities import fig2data

def SaveOptimalAnnotation(Annot, AllPaths, SavePath):
  """
  Saves the optimal annotation to a pickle file
  Should be performed only after multiple cell tracking
  Usually should be performed before interpolating missing cell points, as the interpolated values will be saved as regular (unless this behavior is desired)
  (LZMA compression is used)
  """
  with lzma.open(SavePath, 'wb') as OutFile:
    pickle.dump(Annot,OutFile)
  print("Optimal annotation saved to "+SavePath)

def LoadOptimalAnnotation(LoadPath):
  """
  Loads a previously saved optimal annotation from a pickle file
  The annotation must belong to the given video
  (LZMA compressed file is expected)
  """
  with lzma.open(LoadPath, 'rb') as InFile:
    Annot=pickle.load(InFile)
  AllPaths = AnnotToPaths(Annot)
  print("Optimal annotation loaded from "+LoadPath)
  return Annot, AllPaths

# Prints the values of AllPaths dictionary
def PrintAllPaths(Annot, AllPaths):
  """
  Prints all cell paths
  """
  for path in AllPaths.values():
    for cell in path.values():
      cell.print_data()
  print()

def WriteOptimalAnnotation(Annot, AllPaths, SavePath):
  """
  Saves the optimal annotation to a txt (standard format)
  Should be performed only after multiple cell tracking
  Usually should be performed before interpolating missing cell points, as the interpolated values will be saved as regular (unless this behavior is desired)
  """
  #POS argument got removed
  with open(SavePath, "w") as OutFile:
    OutFile.write("F\tCELLNUM\tX\tY\n")
    for Frame in Annot:
      for CellID in Annot[Frame]:
          contours, hierarchy  = cv2.findContours(np.uint8(np.array(Annot[Frame][CellID])) ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
          for contour in contours:    
            for a in contour:
              l=a[0]
              OutFile.write(str(Frame+1)+"\t"+str(CellID)+"\t"+str(l[0])+"\t"+str(l[1])+"\n")
  print("The optimal annotation is saved in standard txt format to: "+SavePath)

# Displays the cell positions marked with IDs
# DisplayPeriodicity defines which frames will get displayed
def DisplayCellPaths_IDs(Video, Annot, AllPaths,
                         DisplaySegmentation=True, DisplayPeriodicity=1, StartingFrame=0):
  """
  Displays all cell paths as cell IDs on each frame

  - DisplaySegmentation: Boolean variable marking whether to display the segmentations as well
                          If True, non-interpolated segments are displayed as "lime", interpolated ones are displayed as "deepskyblue"
  - DisplayPeriodicity: How frequently should the frames be displayed.
                        Useful for long videos.
                        Minimal possible value=1.
  - Starting Frame: The frame from which the display should be done
                    Can be useful if tracking was only performed after a certain frame
  """
  AnnotFramewiseSum = CalculateAnnotFramewiseSum(Annot)
  for Frame in range(StartingFrame,np.shape(Video)[0]):
    if np.mod(Frame,DisplayPeriodicity)==0:
      print("Frame "+str(Frame))
      fig, (ax1) = plt.subplots(1, 1, figsize=(7, 7))
      ax1.imshow(Video[Frame], cmap=plt.cm.gray, interpolation='nearest')
      if DisplaySegmentation:
        ax1.imshow(AnnotFramewiseSum[Frame], cmap=plt.cm.hot, vmax=2, alpha=.3, interpolation='bilinear')
      for path in AllPaths.values():
        if Frame in path.keys():
          color="lime"
          if path[Frame].interpolated:
            color="deepskyblue"
          ax1.text(path[Frame].x-5, path[Frame].y+5, path[Frame].id, color=color)
      plt.show()

def SaveCellPathsVideo(Video, Annot, AllPaths, VideoPath,
                       Fps = 10, DisplayPeriodicity=1, StartingFrame=None, EndingFrame=None, DisplaySegmentation=True, DisplayFrameNumber=True):
  """
  Saves the cell paths in a similar format as DisplayCellPaths_IDs displays them
  Non-interpolated segments are displayed as "lime", interpolated ones are displayed as "deepskyblue"

  - VideoPath: The path to which the video will be saved
  - DisplaySegmentation: Boolean variable marking whether to display the segmentations
  - DisplaySegmentation: Boolean variable marking whether to display the frame number
  - Fps: The fps of the saved video
  - Starting Frame: The first frame which should be displayed
                    Can be useful if tracking was only performed after a certain frame
  - Ending Frame: The last frame which should be displayed
                  Can be useful if tracking was only performed after a certain frame
  - DisplayPeriodicity: How frequently should the frames be displayed.
                        Useful for long videos.
                        Minimal possible value=1.
  """
  if VideoPath[-4:]!=".mp4":
    raise Exception("Only video paths with mp4 extension are allowed")
  if StartingFrame is None:
    StartingFrame = 1
  if EndingFrame is None:
    EndingFrame = np.shape(Video)[0]+1
  out = cv2.VideoWriter(VideoPath, cv2.VideoWriter_fourcc(*'mp4v'), Fps, (1008,1008), True)
  AnnotFramewiseSum = CalculateAnnotFramewiseSum(Annot)
  for Frame in range(StartingFrame-1,EndingFrame-1):
    if np.mod(Frame,DisplayPeriodicity)==0:
      fig, (ax1) = plt.subplots(1, 1, figsize=(14, 14))
      ax1.imshow(Video[Frame], cmap=plt.cm.gray, interpolation='nearest')
      if DisplaySegmentation:
        ax1.imshow(AnnotFramewiseSum[Frame], cmap=plt.cm.hot, vmax=2, alpha=.3, interpolation='bilinear')
      if DisplayFrameNumber:
        ax1.text(1, 10, "Frame "+str(Frame+1), color="deepskyblue", fontsize=20)
      for path in AllPaths.values():
        if Frame in path.keys():
          color="lime"
          if path[Frame].interpolated:
            color="deepskyblue"
          ax1.text(path[Frame].x-5, path[Frame].y+5, path[Frame].id, color=color, fontsize=20)
      Img = cv2.cvtColor(fig2data(fig), cv2.COLOR_BGRA2RGB)
      out.write(Img)
      plt.close()
  out.release()