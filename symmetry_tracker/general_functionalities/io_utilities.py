import pandas as pd
import json
from pycocotools import mask as coco_mask
from symmetry_tracker.general_functionalities.misc_utilities import CenterMass, BoundingBox

def LoadAnnotJSON(AnnotPath):
  """
  Loads the AnnotDF dataframe from a json
  Initiates centroid and bounding box based on the annotation
  """

  with open(AnnotPath, 'r') as f:
    data = json.load(f)
                         
  AnnotDF = pd.DataFrame(columns = ["Frame", "ObjectID", "SegmentationRLE", "LocalTrackRLE",
                                    "Centroid", "SegBbox", "TrackBbox", "PrevID", "NextID", "TrackID", "Interpolated",
                                    "Class", "AncestorID"])
  
  for Object in data:
    Frame = Object["Frame"]
    FullObjectID = str(Object["ObjectID"])
    IndividualSegImgRLE = Object["SegmentationRLE"]
    Class = str(Object["Class"])
    AncestorID = str(Object["AncestorID"])
    TrackID = str(Object["TrackID"])
    Interpolated = bool(Object["Interpolated"])
    IndividualSegImg = coco_mask.decode(Object["SegmentationRLE"])
    Centroid = CenterMass(IndividualSegImg)
    Bbox = BoundingBox(IndividualSegImg)
    
    AnnotRow = pd.Series({"Frame": Frame, "ObjectID": FullObjectID, "SegmentationRLE": IndividualSegImgRLE,
                      "Centroid": Centroid, "SegBbox":Bbox, "TrackID": TrackID, "Interpolated": Interpolated,
                      "Class": Class, "AncestorID": AncestorID})
    AnnotDF = pd.concat([AnnotDF, AnnotRow.to_frame().T], ignore_index=True)
    
def ExportAnnotJSON(AnnotDF, SavePath):
  """
  Saves the AnnotDF dataframe to a json
  All unnecessary columns for the front-end are removed
  """
  if not SavePath.endswith('.json'):
    raise ValueError("SavePath must have a .json extension")
  AnnotDF_export = AnnotDF[['Frame', 'ObjectID', 'SegmentationRLE', 'TrackID', 'Interpolated', 'Class', 'AncestorID']].copy()
  AnnotDF_export.to_json(SavePath, orient='records')