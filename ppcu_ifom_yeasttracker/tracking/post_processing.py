import numpy as np
import copy
import gc
from scipy.optimize import linear_sum_assignment

from ppcu_ifom_yeasttracker.general_functionalities.misc_utilities import dfs, shift_2d_replace
from ppcu_ifom_yeasttracker.general_functionalities.cell_objects import CellInstance
from ppcu_ifom_yeasttracker.tracking.tracker_utilities import AnnotToPaths
from ppcu_ifom_yeasttracker.tracking.tracker_metrics import SegmentationIOU

def RemoveShortPaths(VideoShape, InputAnnot, InputAllPaths, MinimalPathLength=3):
  """
  Removes the cell tracks from AllPaths and Annot with length<MinimalPathLength

  - MinimalPathLength: Defines the minimal length of a cell path that will not be removed
  """
  Annot = copy.deepcopy(InputAnnot)
  AllPaths = copy.deepcopy(InputAllPaths)
  ShortPathKeys = []
  for path_key in AllPaths.keys():
    if len(AllPaths[path_key].keys())<MinimalPathLength:
      ShortPathKeys.append(path_key)
  for path_key in ShortPathKeys:
    AllPaths.pop(path_key)
    for frame in Annot:
      Annot[frame].pop(path_key,None)
  print("Number of removed cell paths: "+str(len(ShortPathKeys)))
  return VideoShape, Annot, AllPaths

def RemoveBorderPaths(VideoShape, InputAnnot, InputAllPaths, BorderRatio=0.05):
  """
  Removes cell paths that get too close to the image border (at any time)

  - BorderRatio: Gives the ratio that defines closeness to the image edge on all sides in function of the size of the image in that direction
  """
  Annot = copy.deepcopy(InputAnnot)
  AllPaths = copy.deepcopy(InputAllPaths)
  xlim0 = round(VideoShape[1]*BorderRatio)
  xlim1 = round(VideoShape[1]*(1-BorderRatio))
  ylim0 = round(VideoShape[2]*BorderRatio)
  ylim1 = round(VideoShape[2]*(1-BorderRatio))
  BorderPathKeys = []
  for path_key in AllPaths.keys():
    for frame in AllPaths[path_key].keys():
      CellAnnot = np.copy(AllPaths[path_key][frame].annot)
      CellAnnot[xlim0:xlim1,ylim0:ylim1]=0
      if np.sum(CellAnnot!=0):
        BorderPathKeys.append(path_key)
        break
  for path_key in BorderPathKeys:
    AllPaths.pop(path_key)
    for frame in Annot:
      Annot[frame].pop(path_key,None)
  print("Number of removed cell paths: "+str(len(BorderPathKeys)))
  return VideoShape, Annot, AllPaths

def RemoveLargeNewbornCells(VideoShape, InputAnnot, InputAllPaths, MaxNewbornArea = 2000):
  """
  Removes cell paths that start with a too big initial cell size

  - MaxNewbornArea: Maximal size of a newborn cell (area pixel-wise) that is still allowed
  """
  Annot = copy.deepcopy(InputAnnot)
  AllPaths = copy.deepcopy(InputAllPaths)
  FirstFrame = None
  for path_key in AllPaths.keys():
    first_key = min(AllPaths[path_key].keys())
    if FirstFrame is None or first_key<FirstFrame:
      FirstFrame = first_key
  LargeNewbornPathKeys = []
  for path_key in AllPaths.keys():
    first_key = min(AllPaths[path_key].keys())
    if first_key!=FirstFrame and np.sum(AllPaths[path_key][first_key].annot)>MaxNewbornArea:
      LargeNewbornPathKeys.append(path_key)
  for path_key in LargeNewbornPathKeys:
    AllPaths.pop(path_key)
    for frame in Annot:
      Annot[frame].pop(path_key,None)
  print("Number of removed cell paths: "+str(len(LargeNewbornPathKeys)))
  return VideoShape, Annot, AllPaths

def RemoveUnfinishedTracks(VideoShape, InputAnnot, InputAllPaths):
  """
  Removes all paths that do not end on the last frame of tracking
  """
  Annot = copy.deepcopy(InputAnnot)
  AllPaths = copy.deepcopy(InputAllPaths)
  LastFrame = None
  for path_key in AllPaths.keys():
    last_key = max(AllPaths[path_key].keys())
    if LastFrame is None or last_key>LastFrame:
      LastFrame = last_key
  EarlyEndingPathKeys = []
  for path_key in AllPaths.keys():
    last_key = max(AllPaths[path_key].keys())
    if last_key<LastFrame:
      EarlyEndingPathKeys.append(path_key)
  for path_key in EarlyEndingPathKeys:
    AllPaths.pop(path_key)
    for frame in Annot:
      Annot[frame].pop(path_key,None)
  print("Number of removed cell paths: "+str(len(EarlyEndingPathKeys)))
  return VideoShape, Annot, AllPaths

def ManualEquivalence(VideoShape, InputAnnot, InputAllPaths, EquivalencyLists):
  """
  Connects up the cell paths manually defined as equivalent
  All paths will recieve the smallest IDs from the lists
  Like: [[1, 2], [3, 4]] means that paths 1 and 2, and paths 3 and 4 are equivalent (path 1 and 3 are not)
  Be careful to only mark cells as equivalent that do not have common frames!

  - EquivalencyLists: A list of lists where each inner list contains equivalences
  """
  Annot = copy.deepcopy(InputAnnot)
  AllPaths = copy.deepcopy(InputAllPaths)
  NewAnnot = copy.deepcopy(Annot)
  for Ls in EquivalencyLists:
    BaseID = min(Ls)
    print(Ls)
    for Frame in Annot.keys():
      for AnnotID in Annot[Frame]:
        if AnnotID in Ls and AnnotID!=BaseID:
          NewAnnot[Frame][BaseID]=Annot[Frame][AnnotID]
          del NewAnnot[Frame][AnnotID]
  Annot=NewAnnot
  del NewAnnot
  gc.collect()
  AllPaths = AnnotToPaths(Annot)
  print("Manual equivalency transformation successful")
  return VideoShape, Annot, AllPaths

def HeuristicalEquivalence(VideoShape, InputAnnot, InputAllPaths, SimilarityMeasure="IOU", MinRequiredSimilarity=0.2, MaxTimeDiff = 1):
  """
  Connects up the cell paths where frames with maximum MaxTimeDiff temporal difference have a path ensing and starting and the two segmentations are similar
  Similarity is based on SimilarityMeasure, minimal similarity for connection is MinRequiredSimilarity

  - SimilarityMeasure: A keyword for available similarity measures between segmentations, for now only IOU option is available
  - MinRequiredSimilarity: The amount of minimal required similarity based on SimilarityMeasure to connect two cell paths
  - MaxTimeDiff: Maximum allowed temporal difference between loose ends and loose starts of paths (minimum 1 is required ...)
  """
  Annot = copy.deepcopy(InputAnnot)
  AllPaths = copy.deepcopy(InputAllPaths)
  if SimilarityMeasure=="IOU":
    SingleSegmentationSimilarityMeasure=SegmentationIOU
  else:
    raise Exception(SimilarityMeasure + " is not an appropriate keyword for SimilarityMeasure")
  StartingFrame = min(Annot.keys())
  EndingFrame = max(Annot.keys())
  #Defining loose ends and loose starts
  LooseEnds = {}
  LooseStarts = {}
  for Frame in range(StartingFrame,EndingFrame+1):
    LooseEnds[Frame]=[]
    LooseStarts[Frame]=[]
  for CellID in AllPaths:
    PathEnd = max(AllPaths[CellID].keys())
    PathStart = min(AllPaths[CellID].keys())
    if PathEnd!=EndingFrame:
      LooseEnds[PathEnd].append(CellID)
    if PathStart!=StartingFrame:
      LooseStarts[PathStart].append(CellID)
  #Defining equivalency tree-graphs based on the given parameters
  EquivalencyGraph = {}
  for shift in range(1,MaxTimeDiff+1):
    for Frame0 in range(StartingFrame,EndingFrame-shift+1):
      Frame1 = Frame0+shift
      C0_Count = np.shape(LooseEnds[Frame0])[0]
      C1_Count = np.shape(LooseStarts[Frame1])[0]
      if C0_Count!=0 and C1_Count!=0:
        SimilarityMatrix = np.zeros([C0_Count, C1_Count])
        for C0_listID in range(C0_Count):
          for C1_listID in range(C1_Count):
            C0_ID = LooseEnds[Frame0][C0_listID]
            C0_Annot = np.array(Annot[Frame0][C0_ID],dtype='int8')
            C1_ID = LooseStarts[Frame1][C1_listID]
            C1_Annot = np.array(Annot[Frame1][C1_ID],dtype='int8')
            Similarity = SingleSegmentationSimilarityMeasure(C0_Annot, C1_Annot)
            SimilarityMatrix[C0_listID][C1_listID]=Similarity
        SimilarityMatrix = SimilarityMatrix*(SimilarityMatrix>MinRequiredSimilarity)
        try:
          C0_listIDs, C1_listIDs = linear_sum_assignment(1-SimilarityMatrix)
        except:
          print("Error in linear_sum_assignment in HeuristicalEquivalence")
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
  print("Connecting of the paths is finished")
  return VideoShape, Annot, AllPaths

# Linearly interpolates the missing points into a cell path
# If multiple subsequent points are missing, the interpolation is done with an even distribution in between 
# in order by weighting with the time differences
# The interpolated pseudo cell points are marked as "interpolated"
def InterpolateMissingCellPoints(VideoShape, InputAnnot, InputAllPaths):
  """
  Interpolates the missing cell points if there is any
  Should be ran only after a seccessful multiple cell tracking is performed
  Should be performed before building an inheritance tree
  """
  Annot = copy.deepcopy(InputAnnot)
  AllPaths = copy.deepcopy(InputAllPaths)
  for path_key in AllPaths.keys():
    path = AllPaths[path_key]
    FirstFrame = -1
    LastFrame = -1
    for cell in path.values():
      if FirstFrame == -1 or cell.frame < FirstFrame:
        FirstFrame = cell.frame
      if LastFrame == -1 or cell.frame > FirstFrame:
        LastFrame = cell.frame
    for frame in range(FirstFrame, LastFrame+1):
      if frame in path.keys():
        LastOccurence = path[frame]
      else:
        AddFrame = 1
        while not frame+AddFrame in path.keys():
          AddFrame +=1
        NextOccurence = path[frame+AddFrame]
        FrameDiff = NextOccurence.frame-LastOccurence.frame
        InterpX = int(((frame-LastOccurence.frame)*NextOccurence.x+(NextOccurence.frame-frame)*LastOccurence.x)/FrameDiff)
        InterpY = int(((frame-LastOccurence.frame)*NextOccurence.y+(NextOccurence.frame-frame)*LastOccurence.y)/FrameDiff)
        dx = InterpX-LastOccurence.x
        dy = InterpY-LastOccurence.y
        InterpAnnot = shift_2d_replace(LastOccurence.annot, dx, dy, constant=0)
        #plt.imshow(InterpAnnot)
        #plt.show()
        c = CellInstance(InterpX,InterpY,frame,path_key,InterpAnnot,interpolated=True)
        c.define_origin(LastOccurence.origin)
        AllPaths[path_key][frame] = c
        Annot[frame][path_key] = InterpAnnot
  print("Successful interpolation of missing cell points")
  return VideoShape, Annot, AllPaths