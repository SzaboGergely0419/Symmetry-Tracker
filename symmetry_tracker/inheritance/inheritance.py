from symmetry_tracker.general_functionalities.io_utilities import LoadAnnotJSON
import numpy as np

def EuclideanInheritance(AnnotDF, MaxCentroidDistance):
  for child_tr_id in AnnotDF["TrackID"].unique():
    start_frame =  AnnotDF.loc[AnnotDF["TrackID"] == child_tr_id, "Frame"].min()
    if start_frame is not AnnotDF["Frame"].min():
      mother_tr_id = None
      min_distance = None
      AnnotDF_Frame = AnnotDF.loc[AnnotDF["Frame"] == start_frame]
      for mother_canditate_tr_id in AnnotDF_Frame["TrackID"].unique():
        if mother_canditate_tr_id != child_tr_id:
          mother_candidate_ctr = AnnotDF_Frame.loc[AnnotDF_Frame["TrackID"] == mother_canditate_tr_id, "Centroid"].iloc[0]
          child_ctr = AnnotDF_Frame.loc[AnnotDF_Frame["TrackID"] == child_tr_id, "Centroid"].iloc[0]
          distance = np.sqrt((mother_candidate_ctr[0] - child_ctr[0])**2 + (mother_candidate_ctr[1] - child_ctr[1])**2)
          if distance <= MaxCentroidDistance and (min_distance is None or distance < min_distance):
            min_distance = distance
            mother_tr_id = mother_canditate_tr_id
      if mother_tr_id is not None:
        AnnotDF.loc[AnnotDF["TrackID"] == child_tr_id, "AncestorID"] = mother_tr_id
  return AnnotDF

def BuildInheritanceTree(VideoPath, AnnotPath, Method = "Euclidean", MaxCentroidDistance = 60):
  """
  - VideoPath: The path to the video (in .png images format)
  - AnnotPath: The path to a single annotation (segmentation and tracking) belonging to the video at VideoPath
  - Method: The method for inheritance prediction, the current only option is Euclidean
  - MaxCentoidDistance: The maximal allowed distance of centroids for object inheritance assignment
  """

  AnnotDF = LoadAnnotJSON(AnnotPath)
  if len(AnnotDF["TrackID"].unique()) < 1 or (len(AnnotDF["TrackID"].unique()) == 1 and AnnotDF["TrackID"].unique()[0] is None):
    raise Exception(f"Invalid TrackIDs for inheritance detection, run tracking beforehand")
  AnnotDF["AncestorID"] = None

  if Method == "Euclidean":
    AnnotDF = EuclideanInheritance(AnnotDF, MaxCentroidDistance)
  else:
    raise Exception(f"{Method} is not a valid keyword for Method")

  return AnnotDF