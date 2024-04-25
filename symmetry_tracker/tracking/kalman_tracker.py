import cv2
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from symmetry_tracker.tracking.tracker_utilities import LoadAnnotationDF

try:
  from IPython.display import display
  from symmetry_tracker.general_functionalities.misc_utilities import progress
except:
  pass

def EuclideanDistance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def NewKalmanFilter():
  # Define Kalman filter parameters
  kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 states: x, y, dx, dy
  kf.F = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])  # state transition matrix
  kf.H = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])  # measurement function
  kf.R = 10  # measurement uncertainty
  kf.P *= 10  # initial state covariance
  return kf

def KalmanTracking(AnnotDF, MaxCentroidDistance = 20):

  Frames = sorted(AnnotDF['Frame'].unique())
  print("Kalman Tracking")
  try:
    ProgressBar = display(progress(0, len(Frames)), display_id=True)
  except:
    pass

  NewTrackID = 1
  KalmanTracks = {}

  AnnotDF["TrackID"] = None
  for Frame in Frames:
    CurrentObjects = AnnotDF.query("Frame == @Frame")

    if Frame == Frames[0]:
      for ObjectID in CurrentObjects["ObjectID"].unique():
        AnnotDF.loc[AnnotDF["ObjectID"] == ObjectID, "TrackID"] = NewTrackID
        KalmanTracks[NewTrackID] = NewKalmanFilter()
        KalmanTracks[NewTrackID].x[:2] = np.array(AnnotDF.query("ObjectID == @ObjectID")["Centroid"].iloc[0]).reshape(-1, 2, 1)
        KalmanTracks[NewTrackID].update(np.array(AnnotDF.query("ObjectID == @ObjectID")["Centroid"].iloc[0]))
        NewTrackID += 1

    else:

      Objects = []
      for ObjectID in CurrentObjects["ObjectID"].unique():
        Objects.append({"Centroid": AnnotDF.query("ObjectID == @ObjectID")["Centroid"].iloc[0],
                        "ObjectID": ObjectID})

      Predictions = []
      for TrackID in KalmanTracks.keys():
        KalmanTracks[TrackID].predict()
        Predictions.append({"Centroid": KalmanTracks[TrackID].x[:2][:,0],
                            "TrackID": TrackID})

      DistanceMatrix = np.zeros((len(Objects), len(Predictions)))
      for i in range(len(Objects)):
        for j in range(len(Predictions)):
          DistanceMatrix[i, j] = EuclideanDistance(Objects[i]["Centroid"], Predictions[j]["Centroid"])

      ObjPoses, PredPoses = linear_sum_assignment(DistanceMatrix)

      UpdatedTracks = []
      TrackedObjects = []
      for k in range(len(ObjPoses)):
        ObjPos = ObjPoses[k]
        PredPos = PredPoses[k]
        if DistanceMatrix[ObjPos, PredPos] <= MaxCentroidDistance:
          TrackID = Predictions[PredPos]["TrackID"]
          ObjectID = Objects[ObjPos]["ObjectID"]
          NewCentroid = Objects[ObjPos]["Centroid"]
          AnnotDF.loc[AnnotDF["ObjectID"] == ObjectID, "TrackID"] = TrackID
          KalmanTracks[TrackID].update(NewCentroid)
          UpdatedTracks.append(TrackID)
          TrackedObjects.append(ObjectID)

      KalmanTracks = {TrackID: value for TrackID, value in KalmanTracks.items() if TrackID in UpdatedTracks}

      for ObjectID in CurrentObjects["ObjectID"].unique():
        if not ObjectID in TrackedObjects:
          AnnotDF.loc[AnnotDF["ObjectID"] == ObjectID, "TrackID"] = NewTrackID
          KalmanTracks[NewTrackID] = NewKalmanFilter()
          KalmanTracks[NewTrackID].x[:2] = np.array(AnnotDF.query("ObjectID == @ObjectID")["Centroid"].iloc[0]).reshape(-1, 2, 1)
          KalmanTracks[NewTrackID].update(np.array(AnnotDF.query("ObjectID == @ObjectID")["Centroid"].iloc[0]))
          NewTrackID += 1

    try:
      ProgressBar.update(progress(Frame, len(Frames)))
    except:
      pass

  try:
    ProgressBar.update(progress(1, 1))
  except:
    pass

  return AnnotDF

def SingleVideoKalmanTracking(VideoPath, AnnotPath,
                              MinObjectPixelNumber=20, MaxOverlapRatio=0.5,
                              MaxCentroidDistance=20):
  """
  - VideoPath: The path to video in stardard .png images format on which the tracking will be performed
  - AnnotPath: The path to a single annotation (segmentation) belonging to the video at VideoPath
  - MinObjectPixelNumber: Defines the minimal number of pixels in a Object istance for the instance to be recognised as valid
                        Object instances with PixelNumber<MinObjectPixelNumber will be simply deleted during initiation
  - MaxOverlapRatio:  The maximal overlap allowed between annotations in the original annotation.
                      Above MaxOverlapRatio, the area-wise smaller Object will be removed.
                      Not an important parameter if the segmentation is more or less a partitioning
  - MaxCentroidDistance: The maximal distance between estimated and measured centroid for them to be assigned as the same object
  """

  VideoFrames = sorted(os.listdir(VideoPath))
  Img0 = cv2.imread(os.path.join(VideoPath,VideoFrames[0]))
  VideoShape = [len(os.listdir(VideoPath)), np.shape(Img0)[0], np.shape(Img0)[1]]
  AnnotDF = LoadAnnotationDF(AnnotPath, VideoShape, MinObjectPixelNumber, MaxOverlapRatio)
  
  AnnotDF = KalmanTracking(AnnotDF, MaxCentroidDistance)

  return AnnotDF