import numpy as np
import torch
import copy
import gc
from scipy.optimize import linear_sum_assignment

from IPython.display import display
from ppcu_ifom_yeasttracker.general_functionalities.misc_utilities import progress

from ppcu_ifom_yeasttracker.general_functionalities.misc_utilities import CenterMass, dfs
from ppcu_ifom_yeasttracker.tracking.tracker_utilities import *
from ppcu_ifom_yeasttracker.tracking.tracker_metrics import *

def KernelTrack(Video, VideoShape, Model, Device, TimeKernelSize, SegmentationConfidence, CenterPoint, Frame):
  with torch.no_grad():
    inputs = Video[Frame-TimeKernelSize:Frame+TimeKernelSize+1]
    CPImage = np.zeros([VideoShape[1],VideoShape[2]])
    [CMy,CMx]=CenterPoint
    CPImage[CMx-2:CMx+2,CMy-2:CMy+2]=255
    inputs.append(CPImage)
    inputs = np.array(inputs, dtype=float)/255
    inputs = torch.Tensor(np.array(inputs))
    inputs=torch.unsqueeze(inputs, dim=0)

    inputs=inputs.to(torch.device(Device))
    output = np.array(torch.sigmoid(Model(inputs).cpu()))
    output = output>SegmentationConfidence
    output = output*1.0
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)

    torch.cuda.empty_cache()
  return output[0]

def MultipleCellTracking(InputVideo, VideoShape, InputAnnot, Model, Device, TimeKernelSize, SegmentationConfidence, MultiSegmentationSimilarityMeasure,
                         StartingFrame=None, EndingFrame=None, MinRequiredSimilarity=0.5, MaxOverlapRatio=0.5, MaxTimeKernelShift=None):
  """
  Performs multiple cell tracking on the loaded video and annotation (segmentation)

  - StartingFrame: The starting frame for the tracking.
                    The default None means minimal possible value
  - EndingFrame: The ending frame for the tracking. 
                  The default None means maximal possible value
  - MinRequiredSimilarity: The minimal required similarity based on SimilarityMeasure for two trackings to be possibly counted as belonging to the same cell
  - MaxOverlapRatio: The maximal overlap allowed between annotations in the original annotation. 
                      Above MaxOverlapRatio, the area-wise smaller cell will be removed.
                      Not an important parameter if the segmentation is more or less a partitioning
  - MaxTimeKernelShift: The maximal shift allowed between trackings to be recognised as belonging to the same cell
                        Minimal possible value: 1
                        Maximal possible value: 2*TimeKernelSize
                        The default None means maximal possible value
                        Usually None is recommended
                        Smaller values may result in trackings with more "breaks", but possibly fewer errors and slightly faster calculation
  """
  
  Video = copy.deepcopy(InputVideo)
  Annot = copy.deepcopy(InputAnnot)
  #Defining StartingFrame and EndingFrame if they are undefined
  if StartingFrame is None:
    StartingFrame = 0
  if EndingFrame is None:
    EndingFrame = np.shape(Video)[0]-1
  #Defining already used IDs
  UsedIDs = list(Annot[StartingFrame].keys())
  #Extending the Video at the beginning and at the end with repetitive samples to enable full kernel tracking (kind of a boundary condition in time)
  #This means that to later access pos n in video, it must be indexed as pos n+TimeKernelSize
  for ExtraFrame in range(TimeKernelSize):
    Video.insert(0,Video[0])
    Video.append(Video[-1])
  LooseEnds = {}
  LooseStarts = {}
  #Starting multiple cell tracking
  print("Multiple cell tracking with maximal time kernel shift: " +str(MaxTimeKernelShift))
  ProgressBar = display(progress(StartingFrame, EndingFrame), display_id=True)
  for Frame in range(StartingFrame,EndingFrame):
    ProgressBar.update(progress(Frame-StartingFrame, EndingFrame-StartingFrame))
    #Generating kernel tracks for Frame n and n+1
    F0_KernelTracks = {}
    F1_KernelTracks = {}
    F0_Cells = list(Annot[Frame].keys())
    F1_Cells = list(Annot[Frame+1].keys())
    for C0_ID in F0_Cells:
      CellCenter = CenterMass(Annot[Frame][C0_ID])
      F0_KernelTracks[C0_ID]=KernelTrack(Video, VideoShape, Model, Device, TimeKernelSize,
                                         SegmentationConfidence, CellCenter, Frame+TimeKernelSize)
    for C1_ID in F1_Cells:
      CellCenter = CenterMass(Annot[Frame+1][C1_ID])
      F1_KernelTracks[C1_ID]=KernelTrack(Video, VideoShape, Model, Device, TimeKernelSize,
                                         SegmentationConfidence, CellCenter, Frame+TimeKernelSize+1)
    #Creating a similarity matrix with Similarity (ex. IOU) values at appropriate positions and zeros elsewhere 
    #The matrix it is not necessarily rectangular, but that is not an issue
    SimilarityMatrix=np.zeros([max(F0_Cells)+1,max(F1_Cells)+1])
    for C0_ID in F0_Cells:
      for C1_ID in F1_Cells:
        Similarity = MultiSegmentationSimilarityMeasure(F0_KernelTracks[C0_ID],F1_KernelTracks[C1_ID], shift=1)
        SimilarityMatrix[C0_ID][C1_ID]=Similarity
    #Zeroing out the Similarity values smaller than MinRequiredSimilarity in order to not contribute to the linear assignment
    SimilarityMatrix = SimilarityMatrix*(SimilarityMatrix>MinRequiredSimilarity)
    #Calculating the optimal linear assignment between the cell instances (extended Hungarian algorithm)
    #There is a rare error in linear_sum_assignment, which had to be handled by a try block
    #This eliminates all trackings for the given frame, but allows the tracker to continue
    try:
      C0_IDs, C1_IDs = linear_sum_assignment(1-SimilarityMatrix)
    except:
      #WARNING this is not helpful at all!!!
      Annot[Frame+1]={}
      print("Error in linear_sum_assignment at Frame " + str(Frame+1))
      continue
    """
    print("Frame "+str(Frame))
    print(C0_IDs)
    print(C1_IDs)
    plt.matshow(SimilarityMatrix, vmin=0, vmax=1)
    plt.show()
    """
    #Creating the cell paths based on the assignments
    #For successful assignment Similarity (ex. IOU) must be greater than MinRequiredSimilarity
    NewFrameAnnot = {}
    AssignedTargetCellIDs = []
    LooseEnds[Frame]=[]
    LooseStarts[Frame+1]=[]
    for C0_ID in F0_Cells:
      if C0_ID in C0_IDs:
        C0_Pos = int(np.where(C0_IDs == C0_ID)[0])
        OptimalTargetID = C1_IDs[C0_Pos]
        if SimilarityMatrix[C0_ID,OptimalTargetID]>MinRequiredSimilarity:
          NewFrameAnnot[C0_ID]=Annot[Frame+1][OptimalTargetID]
          AssignedTargetCellIDs.append(OptimalTargetID)
        else:
          LooseEnds[Frame].append(C0_ID)
      else:
        LooseEnds[Frame].append(C0_ID)
    #Handling the target cells that did not recieve assignment with new IDs
    for C1_ID in F1_Cells:
      if not (C1_ID in AssignedTargetCellIDs):
        NewID = max(UsedIDs)+1
        UsedIDs.append(NewID)
        NewFrameAnnot[NewID]=Annot[Frame+1][C1_ID]
        LooseStarts[Frame+1].append(NewID)
    #Erasing the redundant (significantly overlapping) annotations
    #For now always the smaller cell segmentation is removed (there might be better solutions than this ...)
    RedundantCellInstances = []
    for An1ID in NewFrameAnnot:
      for An2ID in NewFrameAnnot:
        if An1ID!=An2ID and AnnotationOverlap(NewFrameAnnot[An1ID],NewFrameAnnot[An2ID])>MaxOverlapRatio:
          if np.sum(np.sum(NewFrameAnnot[An1ID]))>=np.sum(np.sum(NewFrameAnnot[An2ID])):
            if not An2ID in RedundantCellInstances:
              RedundantCellInstances.append(An2ID)
          else:
            if not An1ID in RedundantCellInstances:
              RedundantCellInstances.append(An1ID)
    for AnID in RedundantCellInstances:
      del NewFrameAnnot[AnID]
      if AnID in LooseStarts[Frame+1]:
        LooseStarts[Frame+1].remove(AnID)
    #Updating the annotation
    Annot[Frame+1]=NewFrameAnnot
  ProgressBar.update(progress(1, 1))
  #Performing tracking with larger allowed shifts to connect up loose ends and starts if they belong to the same cell
  #The IDs representing the same cells are stored in EquivalencyGraph
  print("Connecting up loose ends if possible")
  EquivalencyGraph = {}
  if MaxTimeKernelShift is None:
    MaxTimeKernelShift=TimeKernelSize*2
  for shift in range(2,MaxTimeKernelShift+1):
    for Frame0 in range(StartingFrame,EndingFrame-shift):
      Frame1 = Frame0+shift
      C0_Count = np.shape(LooseEnds[Frame0])[0]
      C1_Count = np.shape(LooseStarts[Frame1])[0]
      if C0_Count!=0 and C1_Count!=0:
        SimilarityMatrix = np.zeros([C0_Count, C1_Count])
        for C0_listID in range(C0_Count):
          for C1_listID in range(C1_Count):
            C0_ID = LooseEnds[Frame0][C0_listID]
            C0_KernelTrack=KernelTrack(Video, VideoShape, Model, Device, TimeKernelSize,
                                       SegmentationConfidence, CenterMass(Annot[Frame0][C0_ID]),
                                       Frame0+TimeKernelSize)
            C1_ID = LooseStarts[Frame1][C1_listID]
            C1_KernelTrack=KernelTrack(Video, VideoShape, Model, Device, TimeKernelSize,
                                       SegmentationConfidence, CenterMass(Annot[Frame1][C1_ID]),
                                       Frame1+TimeKernelSize)
            Similarity = MultiSegmentationSimilarityMeasure(C0_KernelTrack,C1_KernelTrack, shift=shift)
            SimilarityMatrix[C0_listID][C1_listID]=Similarity
        SimilarityMatrix = SimilarityMatrix*(SimilarityMatrix>MinRequiredSimilarity)
        try:
          C0_listIDs, C1_listIDs = linear_sum_assignment(1-SimilarityMatrix)
        except:
          print("Error in linear_sum_assignment in patch connection")
          continue
        for C0_listID in range(C0_Count):
          if C0_listID in C0_listIDs:
            C0_Pos = int(np.where(C0_listIDs == C0_listID)[0])
            OptimalTargetListID = C1_listIDs[C0_Pos]
            if SimilarityMatrix[C0_listID,OptimalTargetListID]>MinRequiredSimilarity:
              C0_ID = LooseEnds[Frame0][C0_listID]
              C1_ID = LooseStarts[Frame1][OptimalTargetListID]
              if not(C0_ID in EquivalencyGraph.keys()):
                EquivalencyGraph[C0_ID]=[]
              EquivalencyGraph[C0_ID].append(C1_ID)
              if not(C1_ID in EquivalencyGraph.keys()):
                EquivalencyGraph[C1_ID]=[]
              EquivalencyGraph[C1_ID].append(C0_ID)
  #Tarsforming equivalency pairs into equivalency lists using depth-first search
  EquivalencyLists=[]
  for Candidate in EquivalencyGraph.keys():
    CandidateInLists=False
    for Ls in EquivalencyLists:
      if Candidate in Ls:
        CandidateInLists=True
    if not CandidateInLists:
      Visited=set()
      dfs(Visited, EquivalencyGraph, Candidate)
      EquivalencyLists.append(list(Visited))
  #Transforming cells with equivalent IDs to the same ID (lowest ID in the list)
  NewAnnot = copy.deepcopy(Annot)
  for Ls in EquivalencyLists:
    BaseID = min(Ls)
    print(Ls)
    for Frame in range(StartingFrame,EndingFrame+1):
      for AnnotID in Annot[Frame]:
        if AnnotID in Ls and AnnotID!=BaseID:
          NewAnnot[Frame][BaseID]=Annot[Frame][AnnotID]
          del NewAnnot[Frame][AnnotID]
  #Finishing up the tracking (data transformations etc.)
  Annot=NewAnnot
  del NewAnnot
  gc.collect()
  AllPaths = AnnotToPaths(Annot)
  Video=Video[TimeKernelSize:-TimeKernelSize]
  print("Tracking finished!")
  #Returning with the now optimal annotation
  return Annot, AllPaths

def SingleVideoCellTracking(Video, ModelPath, Device, AnnotPath, TimeKernelSize, SimilarityMeasure = "IOU", MinCellPixelNumber=20, SegmentationConfidence = 0.1,
                            StartingFrame=None, EndingFrame=None, MinRequiredSimilarity=0.5, MaxOverlapRatio=0.5, MaxTimeKernelShift=None):
  """
  - Video: The video on which the tracking will be performed
  - ModelPath: The path to the pretrained model (the full model definition, not just the state dictionary)
  - Device: The device on which the segmentator should run (cpu or cuda:0)
  - AnnotPath: The path to a single annotation (segmentation) belonging to the video at VideoPath
  - TimeKernelSize: A constant parameter for the trained Tracker. 
                    TimeKernelSize is the "radius" of the kernel, TimeKernelSize*2+1 is the "diamater" of the actual kernel.
  - SimilarityMeasure: A keyword describing the way of measuring the similarity between overlapping trackings
                        Possible keywords: IOU, DIST, DIST_IOU, LDIST, LDIST_IOU
  - MinCellPixelNumber: Defines the minimal number of pixels in a cell istance for the instance to be recognised as valid
                        Cell instances with PixelNumber<MinCellPixelNumber will be simply deleted during initiation
  - SegmentationConfidence: A number in [0,1] or defining the confidence threshold for the segmentation
                            Lower values are more allowing. Recommanded values are in the [0.05,0.4] range
  - StartingFrame: The starting frame for the tracking.
                  The default None means minimal possible value
  - EndingFrame: The ending frame for the tracking. 
                  The default None means maximal possible value
  - MinRequiredSimilarity: The minimal required similarity based on SimilarityMeasure for two trackings to be possibly counted as belonging to the same cell
  - MaxOverlapRatio: The maximal overlap allowed between annotations in the original annotation. 
                      Above MaxOverlapRatio, the area-wise smaller cell will be removed.
                      Not an important parameter if the segmentation is more or less a partitioning
  - MaxTimeKernelShift: The maximal shift allowed between trackings to be recognised as belonging to the same cell
                        Minimal possible value: 1
                        Maximal possible value: 2*TimeKernelSize
                        The default None means maximal possible value
                        Usually None is recommended
                        Smaller values may result in trackings with more "breaks", but possibly fewer errors and slightly faster calculation
  """
  VideoShape = np.shape(Video)
  Annot = LoadAnnotation(AnnotPath, VideoShape, MinCellPixelNumber)
  MeanCellDiameter = CalculateMeanCellDiameter(Annot)
  Model = LoadPretrainedModel(ModelPath, Device)
  #Defining the cell similarity metric for tracking
  MultiSegmentationSimilarityMeasure = None
  if SimilarityMeasure == "IOU":
    MultiSegmentationSimilarityMeasure = MultiSegmentationIOU
  elif SimilarityMeasure == "DIST":
    MultiSegmentationSimilarityMeasure = MultiSegmentationDIST
  elif SimilarityMeasure == "DIST_IOU":
    MultiSegmentationSimilarityMeasure = MultiSegmentationDISTIOU
  elif SimilarityMeasure == "LDIST":
    MultiSegmentationSimilarityMeasure = MultiSegmentationLDIST
  elif SimilarityMeasure == "LDIST_IOU":
    MultiSegmentationSimilarityMeasure = MultiSegmentationLDISTIOU
  else:
    raise Exception(SimilarityMeasure + " is not an appropriate keyword for SimilarityMeasure")
  
  Annot, AllPaths = MultipleCellTracking(Video, VideoShape, Annot, Model, Device, TimeKernelSize, SegmentationConfidence, MultiSegmentationSimilarityMeasure,
                                         StartingFrame, EndingFrame, MinRequiredSimilarity, MaxOverlapRatio, MaxTimeKernelShift)
  return VideoShape, Annot, AllPaths