import time
import cv2
import os
import numpy as np
import gc
from scipy.optimize import linear_sum_assignment

from symmetry_tracker.general_functionalities.misc_utilities import CenterMass, shift_2d_replace, DecodeMultiRLE
from symmetry_tracker.tracking.tracker_metrics import TracksIOU
from symmetry_tracker.tracking.tracker_utilities import LoadAnnotationDF, LoadPretrainedModel
from symmetry_tracker.tracking.symmetry_tracker import LocalTracking, ConnectedIDReduction

try:
  from IPython.display import display
  from symmetry_tracker.general_functionalities.misc_utilities import progress
except:
  pass

def TracksIOU_ShapeDistance(Array1, Array2, dt):
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

    centroid_diff = np.array(centroids1) - np.array(centroids2)

    rolled_Array2 = np.zeros_like(Array2)
    for (dx, dy), seg2 in zip(centroid_diff, Array2[:-dt]):
        rolled_seg2 = shift_2d_replace(seg2, dx, dy)
        rolled_Array2 += rolled_seg2

    return TracksIOU(Array1, rolled_Array2, dt)

def GlobalAssignment_ShapeDistance(VideoPath, VideoShape, AnnotDF, TimeKernelSize, MinRequiredSimilarity=0.5, MaxTimeKernelShift=None):

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

      SimilarityMatrix = np.zeros((len(T0_IDs),len(Tdt_IDs)))

      for i in range(len(T0_IDs)):
        T0_ID = T0_IDs[i]

        LTR0 = DecodeMultiRLE(AnnotDF.query("ObjectID == @T0_ID")["LocalTrackRLE"].iloc[0])
        for j in range(len(Tdt_IDs)):
          Tdt_ID = Tdt_IDs[j]

          #Removing bbox overlap based elimination as it is "unfair spatial information"
          """
          bbox0 = AnnotDF.query("ObjectID == @T0_ID")["TrackBbox"].iloc[0]
          bboxdt = AnnotDF.query("ObjectID == @Tdt_ID")["TrackBbox"].iloc[0]
          t00 = time.time()
          overlap = BoxOverlap(bbox0, bboxdt)
          if overlap != 0:
          """

          LTRdt = DecodeMultiRLE(AnnotDF.query("ObjectID == @Tdt_ID")["LocalTrackRLE"].iloc[0])
          IOU = TracksIOU_ShapeDistance(LTR0, LTRdt, dt)
          SimilarityMatrix[i, j] = IOU

      SimilarityMatrix = SimilarityMatrix * (SimilarityMatrix >= MinRequiredSimilarity)

      """
      try:
        sns.heatmap(SimilarityMatrix)
        plt.show()
      except:
        pass
      """

      # Hungarian Method based Assignment

      try:
        T0_assignedVals, Tdt_assignedVals = linear_sum_assignment(1-SimilarityMatrix)
      except:
        print(f"Error in linear_sum_assignment at Frame {Frame} to Frame {Frame+dt}")
        continue

      for k in range(len(T0_assignedVals)):
        if SimilarityMatrix[T0_assignedVals[k], Tdt_assignedVals[k]] >= MinRequiredSimilarity:
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

def SingleVideoSymmetryTracking_ShapeDistance(VideoPath, ModelPath, Device, AnnotPath, TimeKernelSize,
                                          Color = "GRAYSCALE", Marker = "CENTROID", MinObjectPixelNumber=20, SegmentationConfidence = 0.1,
                                          MinRequiredSimilarity=0.5, MaxOverlapRatio=0.5, MaxTimeKernelShift=None):
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
  - MinRequiredSimilarity: The minimal required similarity based on IOU for two trackings to be possibly counted as belonging to the same Object
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

  AnnotDF = GlobalAssignment_ShapeDistance(VideoPath, VideoShape, AnnotDF, TimeKernelSize, MinRequiredSimilarity, MaxTimeKernelShift)

  AnnotDF = ConnectedIDReduction(AnnotDF)

  return AnnotDF