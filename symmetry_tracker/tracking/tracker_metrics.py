import numpy as np

def TracksIOU(Array1, Array2, dt):
  if Array1.shape != Array2.shape:
    raise Exception(f"Dimensions of Array1 {Array1.shape} and Array2 {Array2.shape} must be the same")

  Intersection = np.count_nonzero(np.logical_and(Array1[dt:], Array2[:-dt]))
  Union = np.count_nonzero(np.logical_or(Array1[dt:], Array2[:-dt]))

  if Union == 0:
    return 0
  try:
    return Intersection / Union
  except:
    return 0

def SegmentationIOU(Bin1, Bin2):
  if Bin1.shape != Bin2.shape:
    raise Exception(f"Dimensions of Bin1 {Bin1.shape} and Bin2 {Bin2.shape} must be the same")

  Intersection = np.count_nonzero(np.logical_and(Bin1,Bin2))
  Union = np.count_nonzero(np.logical_or(Bin1,Bin2))

  if Union == 0:
    return 0
  try:
    return Intersection / Union
  except:
    return 0

def AnnotationOverlap(Annot1, Annot2):
  if Annot1.shape != Annot2.shape:
    raise Exception(f"Dimensions of Annot1 {Annot1.shape} and Annot2 {Annot2.shape} must be the same")

  Intersection = np.sum(Annot1 * Annot2)
  if Intersection == 0:
    return 0

  SmallerArea = min([np.sum(Annot1), np.sum(Annot2)])

  if SmallerArea == 0:
    return 0
  try:
    return Intersection / SmallerArea
  except:
    return 0