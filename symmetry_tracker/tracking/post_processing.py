import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

from symmetry_tracker.general_functionalities.misc_utilities import shift_2d_replace, BoundingBox
from symmetry_tracker.tracking.tracker_metrics import SegmentationIOU


def RemoveShortPaths(AnnotDF, MinimalPathLength):
  """
  Removes the tracks from AnnotDF with length < MinimalPathLength

  - MinimalPathLength: Defines the minimal length of a cell path that will not be removed
  """
  NumRemovedObjects = 0
  for TrackID in AnnotDF["TrackID"].unique():
    TrackFrames = sorted(AnnotDF.query("TrackID == @TrackID")["Frame"].unique())
    if TrackFrames[-1] - TrackFrames[0] < MinimalPathLength:
      AnnotDF = AnnotDF.query("TrackID != @TrackID")
      NumRemovedObjects += 1
  print(f"Removed {NumRemovedObjects} tracks shorter than {MinimalPathLength} frames")
  return AnnotDF


def HeuristicalEquivalence(AnnotDF, SimilarityMeasure="IOU", MinRequiredSimilarity=0.4, MaxTimeDiff = 1):
  """
  Connects up tracks where frames with maximum MaxTimeDiff temporal difference have a track ending and starting and the two segmentations are similar
  Similarity is based on SimilarityMeasure, minimal similarity for connection is MinRequiredSimilarity
  Should be only performed after tracking but before interpolation
  The method is written in a greedy manner for speed as it is almost impossible that multiple mappable instances are present on the same frame for the same object

  - SimilarityMeasure: A keyword for available similarity measures between segmentations, for now only IOU option is available
  - MinRequiredSimilarity: The amount of minimal required similarity based on SimilarityMeasure to connect two tracks
  - MaxTimeDiff: Maximum allowed temporal difference between loose ends and loose starts of tracks (minimum 1 is required ...)
  """

  if SimilarityMeasure=="IOU":
    SegSimilarityMeasure=SegmentationIOU
  else:
    raise Exception(SimilarityMeasure + " is not an appropriate keyword for SimilarityMeasure")

  OriginalTrackIDs = AnnotDF["TrackID"].unique()
  NumMapped = 0

  for TrackID0 in OriginalTrackIDs:

    # This check is necessary because AnnotDF gets overwritten during the equivalence mapping
    if TrackID0 in AnnotDF["TrackID"].unique():
      TR0LastFrame = max(AnnotDF.query("TrackID == @TrackID0")["Frame"].unique())
      for TrackID1 in OriginalTrackIDs:

        # This check is necessary because AnnotDF gets overwritten during the equivalence mapping
        if TrackID1 in AnnotDF["TrackID"].unique():
          TR1FirstFrame = min(AnnotDF.query("TrackID == @TrackID1")["Frame"].unique())

          if TR1FirstFrame > TR0LastFrame and TR1FirstFrame - TR0LastFrame <= MaxTimeDiff:
            TR0LastInstance = AnnotDF.query("TrackID == @TrackID0 and Frame == @TR0LastFrame").iloc[0].squeeze()
            TR1FirstInstance = AnnotDF.query("TrackID == @TrackID1 and Frame == @TR1FirstFrame").iloc[0].squeeze()

            TR0Seg = coco_mask.decode(TR0LastInstance["SegmentationRLE"]).astype(bool)
            TR1Seg = coco_mask.decode(TR1FirstInstance["SegmentationRLE"]).astype(bool)

            Similarity = SegSimilarityMeasure(TR0Seg, TR1Seg)

            if Similarity >= MinRequiredSimilarity:

              AnnotDF.loc[AnnotDF["TrackID"] == TrackID1, "TrackID"] = TrackID0
              NumMapped += 1

  print(f"Heuristically equivalent mapping of {NumMapped} tracks performed")
  return AnnotDF


def InterpolateMissingObjects(AnnotDF):
  """
  Linearly interpolates the segmentation of missing objects if there is any
  Should be ran only after a seccessful tracking is performed
  Should be performed before building an inheritance tree
  """

  AnnotDF = AnnotDF.query("Interpolated == False")
  NumInterpolated = 0

  for TrackID in AnnotDF["TrackID"].unique():

    TrackFrames = sorted(AnnotDF.query("TrackID == @TrackID")["Frame"].unique())
    FirstFrame = TrackFrames[0]
    LastFrame = TrackFrames[-1]
    LastOccurence = FirstFrame
    for Frame in range(FirstFrame, LastFrame+1):
      if Frame in TrackFrames:
        LastOccurence = Frame
      else:
        AddFrame = 1
        while not Frame+AddFrame in TrackFrames:
          AddFrame += 1
        NextOccurence = Frame+AddFrame
        FrameDiff = NextOccurence-LastOccurence
        LastInstance = AnnotDF.query("TrackID == @TrackID and Frame == @LastOccurence").iloc[0].squeeze()
        NextInstance = AnnotDF.query("TrackID == @TrackID and Frame == @NextOccurence").iloc[0].squeeze()
        InterpX = int(((Frame-LastOccurence)*NextInstance["Centroid"][0]+(NextOccurence-Frame)*LastInstance["Centroid"][0])/FrameDiff)
        InterpY = int(((Frame-LastOccurence)*NextInstance["Centroid"][1]+(NextOccurence-Frame)*LastInstance["Centroid"][1])/FrameDiff)
        Centroid = [InterpX, InterpY]
        dx = InterpX - LastInstance["Centroid"][0]
        dy = InterpY - LastInstance["Centroid"][1]

        LastInstanceSeg = coco_mask.decode(LastInstance["SegmentationRLE"]).astype(np.uint8)
        InterpSegImg = shift_2d_replace(LastInstanceSeg, dx, dy, constant=0)
        InterpSegImgRLE = coco_mask.encode(np.asfortranarray(InterpSegImg))
        InterpSegBbox = BoundingBox(InterpSegImg)
        InterpObjectID = "{:04d}".format(Frame+1)+"{:04d}".format(TrackID)+"ITP"

        AnnotRow = pd.Series({"Frame": Frame, "ObjectID": InterpObjectID, "SegmentationRLE": InterpSegImgRLE,
                      "Centroid": Centroid, "SegBbox": InterpSegBbox, "TrackBbox": None, "PrevID": None, "NextID": None, "TrackID": TrackID, "Interpolated": True})
        AnnotDF = pd.concat([AnnotDF, AnnotRow.to_frame().T], ignore_index=True)

        NumInterpolated+=1

  print(f"Successful interpolation of {NumInterpolated} object instances")
  return AnnotDF
