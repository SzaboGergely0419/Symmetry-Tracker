import numpy as np
import cv2
import torch
import pandas as pd
from pycocotools import mask as coco_mask

from symmetry_tracker.general_functionalities.misc_utilities import CenterMass, BoxOverlap, BoundingBox
from symmetry_tracker.tracking.tracker_metrics import AnnotationOverlap

try:
  from IPython.display import display
  from symmetry_tracker.general_functionalities.misc_utilities import progress
except:
  pass

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

#ZeroFrame can be 0 or 1 (depending on where the counting starts)
def LoadAnnotationDF(AnnotPath, VideoShape, MinObjectPixelNumber = 20, MaxOverlapRatio = 0.2, ZeroFrame = 1, FaultyObjectRemoval = True):
  print("Loading Annotation from:")
  print(AnnotPath)
  AnnotFile = open(AnnotPath, 'r')
  AnnotDF = pd.DataFrame(columns = ["Frame", "ObjectID", "SegmentationRLE", "LocalTrackRLE",
                                    "Centroid", "SegBbox", "TrackBbox", "PrevID", "NextID", "TrackID", "Interpolated"])
  FirstRow = True
  PrevObjectID = -1
  PrevFrame = -1
  PolyLine = []
  NewObjectID = 1
  for line in AnnotFile:
    if not FirstRow and line:
      splitted = line.split()
      Frame = int(splitted[1])
      ObjectID = splitted[2]
      x = int(float(splitted[3]))
      y = int(float(splitted[4]))
      if (ObjectID != PrevObjectID and PrevObjectID!=-1) or (Frame != PrevFrame and PrevFrame != -1):
        IndividualSegImg = np.zeros([VideoShape[1],VideoShape[2]])
        cv2.fillPoly(IndividualSegImg,np.int32([PolyLine]),1)
        IndividualSegImg = np.array(IndividualSegImg, dtype=bool)
        Centroid = CenterMass(IndividualSegImg)
        Bbox = BoundingBox(IndividualSegImg)
        FullObjectID = "{:04d}".format(PrevFrame - ZeroFrame)+"{:04d}".format(NewObjectID)
        IndividualSegImgRLE = coco_mask.encode(np.asfortranarray(IndividualSegImg))
        AnnotRow = pd.Series({"Frame": PrevFrame - ZeroFrame, "ObjectID": FullObjectID, "SegmentationRLE": IndividualSegImgRLE, "LocalTrackRLE": None,
                      "Centroid": Centroid, "SegBbox":Bbox, "TrackBbox": None, "PrevID": None, "NextID": None, "TrackID": None, "Interpolated": False})
        AnnotDF = pd.concat([AnnotDF, AnnotRow.to_frame().T], ignore_index=True)
        PolyLine = []
        NewObjectID += 1
        if Frame != PrevFrame:
          NewObjectID = 1
      PrevObjectID = ObjectID
      PrevFrame = Frame
      PolyLine.append([x,y])
    FirstRow = False

  if FaultyObjectRemoval:
    AnnotDF = RemoveFaultyObjects(AnnotDF, VideoShape, MinObjectPixelNumber, MaxOverlapRatio)

  return AnnotDF

def LoadPretrainedModel(ModelPath, Device):
  Model = torch.load(ModelPath, map_location = torch.device(Device))
  Model.eval()
  print("Model successfully loaded from:")
  print(ModelPath)
  return Model