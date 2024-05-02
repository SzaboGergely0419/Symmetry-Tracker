import time
import cv2
import os
import numpy as np
import gc
from scipy.optimize import linear_sum_assignment

from symmetry_tracker.general_functionalities.misc_utilities import CenterMass, DecodeMultiRLE
from symmetry_tracker.tracking.tracker_utilities import LoadAnnotationDF, LoadPretrainedModel
from symmetry_tracker.tracking.symmetry_tracker import LocalTracking, ConnectedIDReduction

try:
  from IPython.display import display
  from symmetry_tracker.general_functionalities.misc_utilities import progress
except:
  pass


def TracksCentroidL2(Array1, Array2, dt):
    if Array1.shape != Array2.shape:
        raise Exception(f"Dimensions of Array1 {Array1.shape} and Array2 {Array2.shape} must be the same")

    centroids1 = []
    centroids2 = []
    for seg1, seg2 in zip(Array1[dt:], Array2[:-dt]):
        centroid1 = CenterMass(seg1)
        centroid2 = CenterMass(seg2)
        if None in centroid1 or None in centroid2:
            continue
        centroids1.append(centroid1)
        centroids2.append(centroid2)

    distances = [np.linalg.norm(np.array(c1) - np.array(c2)) for c1, c2 in zip(centroids1, centroids2)]
    if len(distances) == 0:
        return float('inf')
    return np.mean(distances)

def GlobalAssignment_L2Distance(VideoPath, VideoShape, AnnotDF, TimeKernelSize, MaxCentroidDistance=20, MaxTimeKernelShift=None):

  VideoFrames = sorted(os.listdir(VideoPath))
  NumFrames = len(VideoFrames)

  if MaxTimeKernelShift is None:
    MaxTimeKernelShift=TimeKernelSize*2

  print("Global Assignment")

  try:
    ProgressBar = display(progress(0, len(AnnotDF)), display_id=True)
  except:
    pass

  AnnotDF[["PrevID", "NextID"]] = None
  for dt in range(1,MaxTimeKernelShift):
    for Frame in range(NumFrames-dt):

      # Similarity Matrix Calculation

      t0 = time.time()

      T0_IDs = AnnotDF.query("Frame == @Frame and NextID.isnull()", engine='python')["ObjectID"].tolist()
      Tdt_IDs = AnnotDF.query("Frame == @Frame+@dt and PrevID.isnull()", engine='python')["ObjectID"].tolist()

      DistanceMatrix = MaxCentroidDistance - np.zeros((len(T0_IDs),len(Tdt_IDs)))

      for i in range(len(T0_IDs)):
        T0_ID = T0_IDs[i]

        LTR0 = DecodeMultiRLE(AnnotDF.query("ObjectID == @T0_ID")["LocalTrackRLE"].iloc[0])
        for j in range(len(Tdt_IDs)):
          Tdt_ID = Tdt_IDs[j]

          #Changing bbox overlap based elimination as it is "unfair shape information"
          bbox0 = np.array(AnnotDF.query("ObjectID == @T0_ID")["TrackBbox"].iloc[0])
          bboxdt = np.array(AnnotDF.query("ObjectID == @Tdt_ID")["TrackBbox"].iloc[0])
          t00 = time.time()

          bbox0_ctr = (bbox0[:2] + bbox0[2:]) / 2
          bboxdt_ctr = (bboxdt[:2] + bboxdt[2:]) / 2
          bbox_distance = np.linalg.norm(bbox0_ctr - bboxdt_ctr)

          if bbox_distance < MaxCentroidDistance:
            LTRdt = DecodeMultiRLE(AnnotDF.query("ObjectID == @Tdt_ID")["LocalTrackRLE"].iloc[0])
            L2Dist = TracksCentroidL2(LTR0, LTRdt, dt)
            DistanceMatrix[i, j] = L2Dist

      """
      sns.heatmap(DistanceMatrix)
      plt.show()
      """

      # Hungarian Method based Assignment

      try:
        T0_assignedVals, Tdt_assignedVals = linear_sum_assignment(DistanceMatrix)
      except:
        print(f"Error in linear_sum_assignment at Frame {Frame} to Frame {Frame+dt}")
        continue

      for k in range(len(T0_assignedVals)):
        if DistanceMatrix[T0_assignedVals[k], Tdt_assignedVals[k]] < MaxCentroidDistance:
          T0_ID = T0_IDs[T0_assignedVals[k]]
          Tdt_ID = Tdt_IDs[Tdt_assignedVals[k]]
          AnnotDF.loc[AnnotDF.query("ObjectID == @T0_ID").index, "NextID"] = Tdt_ID
          AnnotDF.loc[AnnotDF.query("ObjectID == @Tdt_ID").index, "PrevID"] = T0_ID

      t_end = time.time()

      """
      print(f"dt {dt}, Frame {Frame}, t_full {t_end-t0} s")
      """

      try:
        ProgressBar.update(progress(len(AnnotDF.query("not NextID.isnull()")), len(AnnotDF)))
      except:
        pass

  try:
    ProgressBar.update(progress(1, 1))
  except:
    pass

  return AnnotDF

def SingleVideoSymmetryTracking_L2Distance(VideoPath, ModelPath, Device, AnnotPath, TimeKernelSize,
                                           Color = "GRAYSCALE", Marker = "CENTROID", MinObjectPixelNumber=20, SegmentationConfidence = 0.1,
                                           MaxCentroidDistance=20, MaxOverlapRatio=0.5, MaxTimeKernelShift=None):
  """
  - VideoPath: The path to video in stardard .png images format on which the tracking will be performed
  - ModelPath: The path to the pretrained model (the full model definition, not just the state dictionary)
  - Device: The device on which the segmentator should run (cpu or cuda:0)
  - AnnotPath: The path to a single annotation (segmentation) belonging to the video at VideoPath
  - TimeKernelSize: A constant parameter for the trained Tracker.
                    TimeKernelSize is the "radius" of the kernel, TimeKernelSize*2+1 is the "diamater" of the actual kernel.
  - Color: A keyword specific to the used model on the color encoding
           Available options: GRAYSCALE and RGB
  - Marker: A keyword specific to the used model on how the object to be tracked is marked
            Available options: CENTROID and BBOX
  - MinObjectPixelNumber: Defines the minimal number of pixels in a Object istance for the instance to be recognised as valid
                        Object instances with PixelNumber<MinObjectPixelNumber will be simply deleted during initiation
  - SegmentationConfidence: A number in [0,1] or defining the confidence threshold for the segmentation
                            Lower values are more allowing. Recommanded values are in the [0.1,0.9] range
  - MaxCentroidDistance: The maximal (pixel) L2 distance for two trackings to be possibly counted as belonging to the same Object
  - MaxOverlapRatio:  The maximal overlap allowed between annotations in the original annotation.
                      Above MaxOverlapRatio, the area-wise smaller Object will be removed.
                      Not an important parameter if the segmentation is more or less a partitioning
  - MaxTimeKernelShift: The maximal shift allowed between trackings to be recognised as belonging to the same Object
                        Minimal possible value: 1
                        Maximal possible value: 2*TimeKernelSize
                        The default None means maximal possible value
                        Usually None is recommended
                        Smaller values may result in trackings with more "breaks", but possibly fewer errors and slightly faster calculation
  """

  if not Color in ["GRAYSCALE", "RGB"]:
    raise Exception(f"{Color} is an invalid keyword for Color")
  if not Marker in ["CENTROID", "BBOX"]:
    raise Exception(f"{Marker} is not an appropriate keyword for Marker")

  VideoFrames = sorted(os.listdir(VideoPath))
  Img0 = cv2.imread(os.path.join(VideoPath,VideoFrames[0]))
  VideoShape = [len(os.listdir(VideoPath)), np.shape(Img0)[0], np.shape(Img0)[1]]
  AnnotDF = LoadAnnotationDF(AnnotPath, VideoShape, MinObjectPixelNumber, MaxOverlapRatio)

  Model = LoadPretrainedModel(ModelPath, Device)
  AnnotDF = LocalTracking(VideoPath, VideoShape, AnnotDF, Model, Device, TimeKernelSize, Color, Marker, SegmentationConfidence)
  del Model
  gc.collect()

  AnnotDF = GlobalAssignment_L2Distance(VideoPath, VideoShape, AnnotDF, TimeKernelSize, MaxCentroidDistance, MaxTimeKernelShift)

  AnnotDF = ConnectedIDReduction(AnnotDF)

  return AnnotDF