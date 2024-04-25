import time
import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
import gc
from scipy.optimize import linear_sum_assignment

from symmetry_tracker.general_functionalities.misc_utilities import EncodeMultiRLE, DecodeMultiRLE, OuterBoundingBox, BoxOverlap, dfs
from symmetry_tracker.tracking.tracker_metrics import TracksIOU
from symmetry_tracker.tracking.tracker_utilities import LoadAnnotationDF, LoadPretrainedModel

try:
  from IPython.display import display
  from symmetry_tracker.general_functionalities.misc_utilities import progress
except:
  pass


def KernelTrackCentroid(LocalVideo, VideoShape, Model, Device, SegmentationConfidence, ObjectCenter):
  with torch.no_grad():
    inputs = LocalVideo
    CPImage = np.zeros([VideoShape[1],VideoShape[2]])
    [CMy,CMx]=ObjectCenter
    CPImage[CMx-2:CMx+2,CMy-2:CMy+2]=255
    inputs = np.append(inputs, [CPImage], axis=0)
    inputs = np.array(inputs, dtype=float)/255
    inputs = torch.Tensor(np.array(inputs))

    pad_h = (16 - inputs.shape[1] % 16) % 16
    pad_w = (16 - inputs.shape[2] % 16) % 16
    inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode='constant', value=0)

    inputs=torch.unsqueeze(inputs, dim=0)

    inputs=inputs.to(torch.device(Device))
    output = np.array(torch.sigmoid(Model(inputs).cpu()))
    output = output>SegmentationConfidence
    output = output*1.0
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)

    torch.cuda.empty_cache()

    """
    plt.imshow(output[0, 0, :, :])
    plt.show()
    """
  return np.array(output[0], dtype = bool)

def KernelTrackBbox(LocalVideo, VideoShape, Model, Device, SegmentationConfidence, ObjectBbox):
  with torch.no_grad():
    inputs = LocalVideo
    BboxImg = np.zeros([VideoShape[1],VideoShape[2]])
    [x0, y0, x1, y1] = ObjectBbox
    BboxImg[x0:x1,y0:y1]=255
    inputs = np.append(inputs, [BboxImg], axis=0)
    inputs = np.array(inputs, dtype=float)/255
    inputs = torch.Tensor(np.array(inputs))

    pad_h = (16 - inputs.shape[1] % 16) % 16
    pad_w = (16 - inputs.shape[2] % 16) % 16
    inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode='constant', value=0)

    inputs=torch.unsqueeze(inputs, dim=0)

    inputs=inputs.to(torch.device(Device))
    output = np.array(torch.sigmoid(Model(inputs).cpu()))
    output = output>SegmentationConfidence
    output = output*1.0
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)

    torch.cuda.empty_cache()

    """
    plt.imshow(output[0, 0, :, :])
    plt.show()
    """
  return np.array(output[0], dtype = bool)


def LocalTracking(VideoPath, VideoShape, AnnotDF, Model, Device, TimeKernelSize, Color = "GRAYSCALE", Marker = "CENTROID", SegmentationConfidence = 0.2):

  if not Color in ["GRAYSCALE", "RGB"]:
    raise Exception(f"{Color} is an invalid keyword for Color")
  if not Marker in ["CENTROID", "BBOX"]:
    raise Exception(f"{Marker} is not an appropriate keyword for Marker")

  VideoFrames = sorted(os.listdir(VideoPath))
  NumFrames = len(VideoFrames)

  print("Local Tracking")
  try:
    ProgressBar = display(progress(0, NumFrames), display_id=True)
  except:
    pass

  for Frame in range(NumFrames):
    ObjectIDs = AnnotDF.query("Frame == @Frame")["ObjectID"]
    for ObjectID in ObjectIDs:

      # Input image Composition

      if Color == "GRAYSCALE":
        CentralImg = cv2.imread(os.path.join(VideoPath,VideoFrames[Frame]), cv2.IMREAD_GRAYSCALE)
        LocalVideo = np.repeat(CentralImg[np.newaxis, ...], 2*TimeKernelSize+1, axis=0)
        for dt in range(-TimeKernelSize, TimeKernelSize+1):
          if Frame+dt >= 0 and Frame+dt < NumFrames and dt != 0:
            LocalVideo[dt+TimeKernelSize] = cv2.imread(os.path.join(VideoPath,VideoFrames[Frame+dt]), cv2.IMREAD_GRAYSCALE)

      elif Color == "RGB":
        CentralImg = cv2.cvtColor(cv2.imread(os.path.join(VideoPath, VideoFrames[Frame])), cv2.COLOR_BGR2RGB)
        CentralImg = np.transpose(CentralImg, (2,0,1))
        NumReps = 2*TimeKernelSize+1
        LocalVideo = np.zeros((3*NumReps,
                       np.shape(CentralImg)[1],
                       np.shape(CentralImg)[2]),
                      dtype=CentralImg.dtype)
        for Rep in range(NumReps):
          LocalVideo[3*Rep:3*Rep+3] = CentralImg
        for dt in range(-TimeKernelSize, TimeKernelSize+1):
          if Frame+dt >= 0 and Frame+dt < NumFrames and dt != 0:
            LocalImg = cv2.cvtColor(cv2.imread(os.path.join(VideoPath, VideoFrames[Frame+dt])), cv2.COLOR_BGR2RGB)
            LocalVideo[3*(dt+TimeKernelSize):3*(dt+TimeKernelSize)+3] = np.transpose(LocalImg, (2,0,1))

      else:
        raise Exception(f"{Color} is an invalid keyword for Color")

      # Local Tracking

      LocalTrack = None

      if Marker == "CENTROID":
        ObjectCenter = AnnotDF.query("ObjectID == @ObjectID")["Centroid"].iloc[0]
        LocalTrack = KernelTrackCentroid(LocalVideo, VideoShape, Model, Device, SegmentationConfidence, ObjectCenter)

      elif Marker == "BBOX":
        ObjectBbox = AnnotDF.query("ObjectID == @ObjectID")["SegBbox"].iloc[0]
        LocalTrack = KernelTrackBbox(LocalVideo, VideoShape, Model, Device, SegmentationConfidence, ObjectBbox)

      AnnotDF.loc[AnnotDF.query("ObjectID == @ObjectID").index, "LocalTrackRLE"] = [EncodeMultiRLE(LocalTrack)]

      # 3D Boundary Box calculation

      bbox = OuterBoundingBox(LocalTrack)
      AnnotDF.loc[AnnotDF.query("ObjectID == @ObjectID").index, "TrackBbox"] = [bbox]

    try:
      ProgressBar.update(progress(Frame, NumFrames))
    except:
      pass

  try:
    ProgressBar.update(progress(1, 1))
  except:
    pass

  return AnnotDF


def GlobalAssignment(VideoPath, VideoShape, AnnotDF, TimeKernelSize, MinRequiredSimilarity=0.5, MaxTimeKernelShift=None):

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

          bbox0 = AnnotDF.query("ObjectID == @T0_ID")["TrackBbox"].iloc[0]
          bboxdt = AnnotDF.query("ObjectID == @Tdt_ID")["TrackBbox"].iloc[0]
          t00 = time.time()
          overlap = BoxOverlap(bbox0, bboxdt)
          if overlap != 0:
            LTRdt = DecodeMultiRLE(AnnotDF.query("ObjectID == @Tdt_ID")["LocalTrackRLE"].iloc[0])
            IOU = TracksIOU(LTR0, LTRdt, dt)
            SimilarityMatrix[i, j] = IOU

      SimilarityMatrix = SimilarityMatrix * (SimilarityMatrix >= MinRequiredSimilarity)

      """
      sns.heatmap(SimilarityMatrix)
      plt.show()
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


def ConnectedIDReduction(AnnotDF):

  AnnotDF["TrackID"] = None

  # Transforming key-value pair matches to an undirected graph dictionary

  EquivalencyGraph = {}
  for ObjectID in AnnotDF["ObjectID"].unique():
    Neighbors = set()
    PrevID = AnnotDF.query("ObjectID == @ObjectID")["PrevID"].iloc[0]
    NextID = AnnotDF.query("ObjectID == @ObjectID")["NextID"].iloc[0]
    if not PrevID is None:
      Neighbors.add(PrevID)
    if not NextID is None:
      Neighbors.add(NextID)
    EquivalencyGraph[ObjectID] = Neighbors

  # Creating equivalency sets (everything that is connected is equivavlent)

  EquivalencySets = []
  for RootCandidate in EquivalencyGraph.keys():
    CandidateInEqSets=False
    for EqSet in EquivalencySets:
      if RootCandidate in EqSet:
        CandidateInEqSets=True
    if not CandidateInEqSets:
      Visited=set()
      dfs(Visited, EquivalencyGraph, RootCandidate)
      EquivalencySets.append(Visited)

  # Generating new (minimal) IDs into the AnnotDF

  NewTrackID = 1
  for EquivalentIDs in EquivalencySets:
    for ObjectID in EquivalentIDs:
      AnnotDF.loc[AnnotDF.query("ObjectID == @ObjectID").index, "TrackID"] = NewTrackID

    NewTrackID += 1

  return AnnotDF


def SingleVideoSymmetryTracking(VideoPath, ModelPath, Device, AnnotPath, TimeKernelSize,
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

  AnnotDF = GlobalAssignment(VideoPath, VideoShape, AnnotDF, TimeKernelSize, MinRequiredSimilarity, MaxTimeKernelShift)

  AnnotDF = ConnectedIDReduction(AnnotDF)

  return AnnotDF