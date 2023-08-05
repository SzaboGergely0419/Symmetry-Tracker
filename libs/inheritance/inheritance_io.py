import numpy as np
import matplotlib.pyplot as plt
import cv2

from general_functionalities.misc_utilities import fig2data
from tracking.tracker_utilities import CalculateAnnotFramewiseSum

def PrintInheritanceTree(InheritanceTree):
  """
  Prints the interitance tree
  Should be only performed after building the inheritance tree
  """
  for InheritanceCell in InheritanceTree:
    InheritanceCell.print_data()
  print()

def WriteInheritanceTree(InheritanceTree, SavePath):
  """
  Writes the interitance tree to a standard format txt
  Value None at Mothers means that the cell has no mother
  Should be only performed after building the inheritance tree
  """
  NoMotherValue = None
  with open(SavePath, "w") as OutFile:
    OutFile.write("Cell_ID\tFirstFrame\tLastFrame\tMother\n")
    for InheritanceCell in InheritanceTree:
      LastFrame = max(list(InheritanceCell.full_path.keys()))+1
      if len(InheritanceCell.parents)==0:
          OutFile.write(str(InheritanceCell.origin.id)+"\t"+
                      str(InheritanceCell.origin.frame+1)+"\t"+
                      str(LastFrame)+"\t"+
                      str(NoMotherValue)+"\n")
      else:
        for Mother in InheritanceCell.parents:
          OutFile.write(str(InheritanceCell.origin.id)+"\t"+
                        str(InheritanceCell.origin.frame+1)+"\t"+
                        str(LastFrame)+"\t"+
                        str(Mother.origin.id)+"\n")
  print("Inheritance tree is saved in standard txt format to: "+SavePath)

# Displays the inheritance tree onto the last frame
# The cells will be connected by their last occurence positions
def DisplayInheritanceTree(Video, InheritanceTree):
  """
  Displays the inheritance tree onto the last frame with arrows
  Should be only performed after building the inheritance tree
  Only useful for recordings with few cells, otherwise the result will be very crowded
  """
  fig, (ax1) = plt.subplots(1, 1, figsize=(7, 7))
  ax1.imshow(Video[-1], cmap=plt.cm.gray)
  for InheritanceCell in InheritanceTree:
    ICLastKey = np.max(list(InheritanceCell.full_path.keys()))
    ICLastFrame =  InheritanceCell.full_path[ICLastKey]
    color = (1,.5,0)
    if not InheritanceCell.parents:
      c = plt.Circle((ICLastFrame.x,ICLastFrame.y),4,color=color, fill=True)
      ax1.add_patch(c)
    for Child in InheritanceCell.children:
      CLastKey = np.max(list(Child.full_path.keys()))
      CLastFrame = Child.full_path[CLastKey]
      ax1.arrow(ICLastFrame.x,ICLastFrame.y,CLastFrame.x-ICLastFrame.x,CLastFrame.y-ICLastFrame.y,color=color,head_width=5,head_length=10,length_includes_head=True)
  plt.show()

def SaveAllCellDataVideo(Video, Annot, AllPaths, InheritanceTree,
                          VideoPath, Fps = 10, DisplayPeriodicity=1, StartingFrame=None, EndingFrame=None,
                          DisplaySegmentation=True, DisplayFrameNumber=True, DisplayCellIDs = True, DisplayInheritances = None):
  """
  Saves the cell paths in a similar format as DisplayCellPaths_IDs displays them

  - VideoPath: The path to which the video will be saved
  - DisplayCellIDs: Boolean variable marking whether to display the cell IDs
                    If True, non-interpolated segments are displayed as "lime", interpolated ones are displayed as "deepskyblue"
  - DisplaySegmentation: Boolean variable marking whether to display the segmentations
  - DisplaySegmentation: Boolean variable marking whether to display the frame number
  - DisplayInheritances: Keyword variable marking whether to display the inheritances as arrows
                          Valid keywords: None, "PERMA", "FLASH"
                          Default value is None
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
      if DisplayCellIDs:
        for path in AllPaths.values():
          if Frame in path.keys():
            color="lime"
            if path[Frame].interpolated:
              color="deepskyblue"
            ax1.text(path[Frame].x-5, path[Frame].y+5, path[Frame].id, color=color, fontsize=20)
      if not(DisplayInheritances is None):
        for MotherCell in InheritanceTree:
          MotherID = MotherCell.origin.id
          for ChildCell in MotherCell.children:
            ChildFrame = np.min(list(ChildCell.full_path.keys()))
            ChildID = ChildCell.origin.id
            if DisplayInheritances == "FLASH":
              if Frame == ChildFrame:
                MotherInst = AllPaths[MotherID][Frame]
                ChildInst = AllPaths[ChildID][Frame]
                ax1.arrow(MotherInst.x,MotherInst.y,ChildInst.x-MotherInst.x,ChildInst.y-MotherInst.y,
                          color="fuchsia",head_width=8,head_length=10,length_includes_head=True)
            if DisplayInheritances == "PERMA":
              if Frame in AllPaths[MotherID] and Frame in AllPaths[ChildID]:
                MotherInst = AllPaths[MotherID][Frame]
                ChildInst = AllPaths[ChildID][Frame]
                ax1.arrow(MotherInst.x,MotherInst.y,ChildInst.x-MotherInst.x,ChildInst.y-MotherInst.y,
                          color="fuchsia",head_width=8,head_length=10,length_includes_head=True)
      Img = cv2.cvtColor(fig2data(fig), cv2.COLOR_BGRA2RGB)
      out.write(Img)
      plt.close()
  out.release()