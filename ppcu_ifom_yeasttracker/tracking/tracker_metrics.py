"""
Warning! MeanCellDiameter is currently not calculated, 
therefore MultiSegmentationLDIST and MultiSegmentationLDISTIOU
are not usable
"""

import numpy as np
from scipy.stats import gmean
from scipy.special import expit

from ppcu_ifom_yeasttracker.general_functionalities.misc_utilities import CenterMass

def CalculateMeanCellDiameter(Annot):
  """
  Robust mean of cell diameters (based on the assuption that a cell is roughly circular)
  The existance of valid Annot is required for the calculation
  """
  Diameters = []
  for Frame in Annot.keys():
    for CellID in Annot[Frame].keys():
      Cell = Annot[Frame][CellID]
      CellArea = np.sum(Cell)
      CellDiameter = np.sqrt(CellArea/3.1415)*2
      Diameters.append(CellDiameter)
  MeanDiameter = gmean(Diameters)
  return MeanDiameter

def SegmentationIOU(Bin1, Bin2):
  Intersection = Bin1*Bin2
  Union = (Bin1+Bin2)-Intersection
  return np.nan_to_num(np.sum(Intersection)/np.sum(Union), nan=0.0, posinf=0.0, neginf=0.0)

def MultiSegmentationCenterMasses(Array):
  Centres = []
  for Img in Array:
    [Cx, Cy] = CenterMass(Img)
    Centres.append(np.array([Cx, Cy]))
  return np.array(Centres)

def MultiSegmentationDIST(Array1,Array2,shift):
  """
  Calculates the similarity between Array1 and Array2 based on eulidean distance and the max possible distance on the image
  This is a similarity measure in the range [0,1], closer cells will give higher numbers
  If there is an error due to undefineable cell center (can happen due to the kernel tracker NN), the instance will not be considered
  """
  if(np.shape(Array1)!=np.shape(Array2)): 
    raise Exception("Dimensions of Array1 "+np.str(np.shape(Array1))+" and Array2 "+np.str(np.shape(Array2))+" must be the same" )
  SubArray1 = Array1[shift:]
  SubArray2 = Array2[:-shift]
  Centres1 = MultiSegmentationCenterMasses(SubArray1)
  Centres2 = MultiSegmentationCenterMasses(SubArray2)
  BadIndexes = []
  for idx in range(np.shape(Centres1)[0]):
    if np.any(Centres1[idx]==None) or np.any(Centres2[idx]==None):
      BadIndexes.append(idx)
  Centres1 = np.delete(Centres1, BadIndexes, 0)
  Centres2 = np.delete(Centres2, BadIndexes, 0)
  if Centres1.size==0 or Centres2.size==0:
    return 0
  Distances = np.sqrt(np.array(np.sum(np.power(Centres1-Centres2,2),axis=1)).astype(float))
  [N,H,W] = np.shape(SubArray1)
  MaxDistSum = np.sqrt(H*H+W*W)*N
  Similarity = 1-np.sum(Distances)/MaxDistSum
  return Similarity

def MultiSegmentationLDIST(Array1,Array2,shift):
  """
  Calculates the similarity between Array1 and Array2 based on a logistic function on the eulidean distance and the mean cell diameter
  This is a similarity measure in the range [0,1], closer cells will give higher numbers, but the max possible value is a bit lower than 1
  (In case of c1=4, it is 0.98)
  If there is an error due to undefineable cell center (can happen due to the kernel tracker NN), the instance will not be considered
  """
  if(np.shape(Array1)!=np.shape(Array2)): 
    raise Exception("Dimensions of Array1 "+np.str(np.shape(Array1))+" and Array2 "+np.str(np.shape(Array2))+" must be the same" )
  SubArray1 = Array1[shift:]
  SubArray2 = Array2[:-shift]
  Centres1 = MultiSegmentationCenterMasses(SubArray1)
  Centres2 = MultiSegmentationCenterMasses(SubArray2)
  BadIndexes = []
  for idx in range(np.shape(Centres1)[0]):
    if np.any(Centres1[idx]==None) or np.any(Centres2[idx]==None):
      BadIndexes.append(idx)
  Centres1 = np.delete(Centres1, BadIndexes, 0)
  Centres2 = np.delete(Centres2, BadIndexes, 0)
  if Centres1.size==0 or Centres2.size==0:
    return 0
  Distances = np.sqrt(np.array(np.sum(np.power(Centres1-Centres2,2),axis=1)).astype(float))
  LinearSimilarity = 1-np.sum(Distances)/MeanCellDiameter
  #https://www.wolframalpha.com/input?i=1%2F%281%2Be%5E%28-%288x-4%29%29%29
  c1 = 4
  LogisticSimilarity = expit(LinearSimilarity*2*c1-c1)
  return LogisticSimilarity

def MultiSegmentationIOU(Array1,Array2,shift):
  if(np.shape(Array1)!=np.shape(Array2)): 
    raise Exception("Dimensions of Array1 "+np.str(np.shape(Array1))+" and Array2 "+np.str(np.shape(Array2))+" must be the same" )
  SubArray1 = Array1[shift:]
  SubArray2 = Array2[:-shift]
  Intersection = SubArray1*SubArray2
  Union = (SubArray1+SubArray2)-Intersection
  return np.nan_to_num(np.sum(Intersection)/np.sum(Union), nan=0.0, posinf=0.0, neginf=0.0)

def MultiSegmentationDISTIOU(Array1,Array2,shift):
  DistSimilarity = MultiSegmentationDIST(Array1,Array2,shift)
  IOU = MultiSegmentationIOU(Array1,Array2,shift)
  return (DistSimilarity+IOU)/2

def MultiSegmentationLDISTIOU(Array1,Array2,shift):
  DistSimilarity = MultiSegmentationLDIST(Array1,Array2,shift)
  IOU = MultiSegmentationIOU(Array1,Array2,shift)
  return (DistSimilarity+IOU)/2

#Calculating the intersection over the area of the smaller cell (similar to IOU, but differently sensitive to different sized cells)
def AnnotationOverlap(Annot1,Annot2):
  if(np.shape(Annot1)!=np.shape(Annot2)):
    raise Exception("Dimensions of Image 1 "+np.str(np.shape(Annot1))+" and Image 2 "+np.str(np.shape(Annot2))+" must be the same" )
  H, W = np.shape(Annot1)
  Intersection = np.sum(Annot1*Annot2)
  SmallerArea = min([np.sum(Annot1),np.sum(Annot2)])
  return Intersection/SmallerArea