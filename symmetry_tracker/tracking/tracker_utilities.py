import numpy as np
import pandas as pd
import torch
from pycocotools import mask as coco_mask

from symmetry_tracker.general_functionalities.misc_utilities import BoxOverlap
from symmetry_tracker.tracking.tracker_metrics import AnnotationOverlap

try:
  from IPython.display import display
  from symmetry_tracker.general_functionalities.misc_utilities import progress
except:
  pass

def TransformToTrackingAnnot(AnnotDF):
  """
  Properely initializes the None columns (LocalTrackRLE, TrackBbox, PrevID, NextID, TrackID) of the AnnotDF 
  """
  TRAnnotDF = pd.DataFrame(columns = ["Frame", "ObjectID", "SegmentationRLE", "LocalTrackRLE",
                                    "Centroid", "SegBbox", "TrackBbox", "PrevID", "NextID", "TrackID", "Interpolated",
                                    "Class", "AncestorID"])
  
  for _,AnnotRow in AnnotDF.iterrows():
    TRAnnotRow = pd.Series({"Frame": AnnotRow["Frame"], 
                            "ObjectID": AnnotRow["ObjectID"],
                            "SegmentationRLE": AnnotRow["SegmentationRLE"],
                            "LocalTrackRLE": None, 
                            "Centroid": AnnotRow["Centroid"], 
                            "SegBbox": AnnotRow["SegBbox"], 
                            "TrackBbox": None, 
                            "PrevID": None, 
                            "NextID": None, 
                            "TrackID": None, 
                            "Interpolated": False,
                            "Class": AnnotRow["Class"], 
                            "AncestorID": AnnotRow["AncestorID"]})
    TRAnnotDF = pd.concat([TRAnnotDF, TRAnnotRow.to_frame().T], ignore_index=True)

  return TRAnnotDF

def RemoveFaultyObjects(AnnotDF, VideoShape, MinObjectPixelNumber, MaxOverlapRatio):
  """
  Removes the objects from the annotation, which
  - have too small size based on self.MinObjectPixelNumber
  - have no defineable center
  """
  print("Removing faulty object instances")

  Counter = 0
  FaultyInstances = {}

  NumFrames = len(AnnotDF["Frame"].unique())
  try:
    ProgressBar = display(progress(0, NumFrames), display_id=True)
  except:
    pass

  for Frame in AnnotDF["Frame"].unique():

    try:
      ProgressBar.update(progress(Frame, NumFrames))
    except:
      pass

    FaultyInstances[Frame]=[]
    ObjectIDs = AnnotDF.query("Frame == @Frame")["ObjectID"]

    for ObjectID in ObjectIDs:
      ObjectBbox = AnnotDF.query("ObjectID == @ObjectID")["SegBbox"].iloc[0]
      ObjectSeg = coco_mask.decode(AnnotDF.query("ObjectID == @ObjectID")["SegmentationRLE"].iloc[0])
      Size = np.sum(ObjectSeg)
      [Cx,Cy] = AnnotDF.query("ObjectID == @ObjectID")["Centroid"].iloc[0]

      HasBetterCoverage = False
      for Object2ID in ObjectIDs:
        if ObjectID != Object2ID:
          Object2Bbox = AnnotDF.query("ObjectID == @Object2ID")["SegBbox"].iloc[0]

          if BoxOverlap(ObjectBbox, Object2Bbox) > 0:
            Object2Seg = coco_mask.decode(AnnotDF.query("ObjectID == @Object2ID")["SegmentationRLE"].iloc[0])
            if np.sum(ObjectSeg) <= np.sum(Object2Seg) and AnnotationOverlap(ObjectSeg, Object2Seg) > MaxOverlapRatio:
              HasBetterCoverage = True

      if Size<MinObjectPixelNumber or [Cx, Cy]==[None,None] or Cx<=0 or Cy<=0 or Cx>=VideoShape[2]-1 or Cy>=VideoShape[1]-1 or HasBetterCoverage:
        FaultyInstances[Frame].append(ObjectID)
        Counter+=1

  try:
    ProgressBar.update(progress(1, 1))
  except:
    pass

  for Frame in FaultyInstances:
    for ObjectID in FaultyInstances[Frame]:
      AnnotDF = AnnotDF.query("ObjectID != @ObjectID")

  print(f"Number of removed faulty object instances: {Counter}")
  return AnnotDF

def LoadPretrainedModel(ModelPath, Device):
  Model = torch.load(ModelPath, map_location = torch.device(Device))
  Model.eval()
  print("Model successfully loaded from:")
  print(ModelPath)
  return Model