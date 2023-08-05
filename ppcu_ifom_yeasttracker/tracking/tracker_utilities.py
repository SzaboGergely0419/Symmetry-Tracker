import numpy as np
import cv2
import torch

from ppcu_ifom_yeasttracker.general_functionalities.misc_utilities import CenterMass
from ppcu_ifom_yeasttracker.general_functionalities.cell_objects import CellInstance

def CalculateAnnotFramewiseSum(Annot):
  AnnotFramewiseSum = {}
  for Frame in Annot:
    AnnotSumImg = np.sum(np.array(list(Annot[Frame].values())), axis=0)>0
    AnnotFramewiseSum[Frame]=AnnotSumImg
  return AnnotFramewiseSum

def RemoveFaultyCells(Annot, VideoShape, MinCellPixelNumber):
  """
  Removes the cells from the annotation, which 
  - have too small size based on self.MinCellPixelNumber
  - have no defineable center
  """
  Counter = 0
  FaultyInstances = {}
  for Frame in Annot:
    FaultyInstances[Frame]=[]
    for CellID in Annot[Frame]:
      Size = np.sum(Annot[Frame][CellID])
      [Cx,Cy] = CenterMass(Annot[Frame][CellID])
      if Size<MinCellPixelNumber or [Cx, Cy]==[None,None] or Cx<=0 or Cy<=0 or Cx>=VideoShape[1]-1 or Cy>=VideoShape[2]-1:
        FaultyInstances[Frame].append(CellID)
        Counter+=1
  for Frame in FaultyInstances:
    for CellID in FaultyInstances[Frame]:
      Annot[Frame].pop(CellID)
  print("Number of removed faulty cell instances: "+str(Counter))
  return Annot

#ZeroFrame can be 0 or 1 (depending on where the counting starts)
def LoadAnnotation(AnnotPath, VideoShape, MinCellPixelNumber, ZeroFrame=1):
  AnnotFile = open(AnnotPath, 'r')
  Annot = {}
  FirstRow = True
  PrevCellID = -1
  PrevFrame = -1
  PolyLine = []
  SingleFrameAnnotations = {}
  for line in AnnotFile:
    if not FirstRow and line:
      splitted = line.split()
      frame = int(splitted[1])
      cellID = int(splitted[2])
      x = int(float(splitted[3]))
      y = int(float(splitted[4]))
      if frame != PrevFrame and PrevFrame != -1:
        Annot[PrevFrame-ZeroFrame]=SingleFrameAnnotations
        SingleFrameAnnotations = {}
      if cellID != PrevCellID and PrevCellID!=-1:
        IndividualAnnotImg = np.zeros([VideoShape[1],VideoShape[2]])
        cv2.fillPoly(IndividualAnnotImg,np.int32([PolyLine]),1)
        IndividualAnnotImg = np.array(IndividualAnnotImg, dtype=bool)
        SingleFrameAnnotations[cellID]=IndividualAnnotImg
        PolyLine = []
      PrevCellID = cellID
      PrevFrame = frame
      PolyLine.append([x,y])
    FirstRow = False
  Annot[PrevFrame-ZeroFrame]=SingleFrameAnnotations
  Annot = RemoveFaultyCells(Annot, VideoShape, MinCellPixelNumber)
  print("Annotation loaded from "+AnnotPath)
  return Annot

def LoadPretrainedModel(ModelPath, Device):
  Model = torch.load(ModelPath, map_location = torch.device(Device))
  Model.eval()
  print("Model successfully loaded from:")
  print(ModelPath)
  return Model

def AnnotToPaths(Annot):
  AllPaths = {}
  for Frame in Annot:
    for CellID in Annot[Frame].keys():
      [cx, cy] = CenterMass(Annot[Frame][CellID])
      CellInst = CellInstance(cx, cy, Frame, CellID, Annot[Frame][CellID])
      if CellID in AllPaths.keys():
        AllPaths[CellID][Frame]=CellInst
      else:
        CellPath = {}
        CellPath[Frame]=CellInst
        AllPaths[CellID]=CellPath
  for CellID in AllPaths:
    FirstOccurence = min(AllPaths[CellID].keys())
    for Frame in AllPaths[CellID]:
      AllPaths[CellID][Frame].define_origin(AllPaths[CellID][FirstOccurence])
  return AllPaths