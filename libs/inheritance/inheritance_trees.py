import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import route_through_array

from general_functionalities.cell_objects import CellInheritance
from tracking.tracker_utilities import CalculateAnnotFramewiseSum

# Euclidean solution
def BuildInheritanceTree_Euclidean(Annot, AllPaths, MaxInheritanceDistance = 50):
  """
  Builds an inheritance tree between cell tracks (mother-daughter relationship) using euclidean distance
  Baseline solution, usually good, but not realisic
  Should be performed only after multiple cell tracking and interpolation of missing cell points

  - MaxInheritanceDistance: The maximal distance between the cell centers (pixel-wise) in which the inheritance between the cells is possible
  """
  InheritanceTree = []
  for path_key in AllPaths.keys():
    path = AllPaths[path_key]
    FirstFrame = -1
    for cell in path.values():
      if FirstFrame == -1 or cell.frame < FirstFrame:
        FirstFrame = cell.frame
    ci = CellInheritance(path[FirstFrame].origin, path)
    InheritanceTree.append(ci)
  for path_key in AllPaths.keys():
    path = AllPaths[path_key]
    FirstFrame = -1
    for cell in path.values():
      if FirstFrame == -1 or cell.frame < FirstFrame:
        FirstFrame = cell.frame
    if FirstFrame != 0:
      MinInheritDistSquared = -1
      AncestorCell = None
      for path_origin in AllPaths.values():
        if FirstFrame in path_origin.keys() and path_origin[FirstFrame].id != path[FirstFrame].id:
          InheritDistSquared = np.power(path_origin[FirstFrame].x-path[FirstFrame].x,2)+ \
                                np.power(path_origin[FirstFrame].y-path[FirstFrame].y,2)
          if InheritDistSquared < MaxInheritanceDistance*MaxInheritanceDistance and \
              (MinInheritDistSquared == -1 or InheritDistSquared < MinInheritDistSquared):
            AncestorCell = path_origin[FirstFrame].origin
            MinInheritDistSquared = InheritDistSquared
      if AncestorCell != None:
        for CurrentCellPos in range(np.shape(InheritanceTree)[0]):
          if InheritanceTree[CurrentCellPos].origin.id == path[FirstFrame].origin.id:
            for ACell in InheritanceTree:
              if ACell.origin.id == AncestorCell.id:
                InheritanceTree[CurrentCellPos].add_parent(ACell)
          if InheritanceTree[CurrentCellPos].origin.id == AncestorCell.id:
            for CCell in InheritanceTree:
              if CCell.origin.id == path[FirstFrame].origin.id:
                InheritanceTree[CurrentCellPos].add_child(CCell)
  print("Inheritance tree successfully built")
  return InheritanceTree

# Route finding solution
# BlackEpsilon is a value offset on the image to avoid completely black pixels (which would break the algorithm due to having 0 weight)
# Can be length based (the length of the minimum path is taken into consideration) or can be weight based (weights over the minimal path)
# Length based is better if there are non removed obstacles inside the cell, or if there are holes on the cell wall
# MaxWeight is a parameter which defines whether the path can cross some lighter pixels or not (larger values allow more crossing)
# MaxInheritanceDistance defines the maximal possible distance for inheritance in pixels (to lower computational needs)
def BuildInheritanceTree_RouteFinding(Video, Annot, AllPaths, DisplayMethodResults = False, BlackEpsilon = 0.1, MaxInheritanceDistance = 50, MaxWeight = 500, WeightBased = False):
  """
  Builds an inheritance tree between cell tracks (mother-daughter relationship) using optimal route finding between the cells
  A solution inspired by the possibility of fluid movement between cells
  Should be performed only after multiple cell tracking and interpolation of missing cell points

  - DisplayMethodResults: A debugging and validation tool to display the method results
  - BlackEpsilon: The minimal weight for a pixel value during the route finding
  - MaxInheritanceDistance: The maximal distance between the cell centers (pixel-wise) in which the inheritance between the cells is possible
  - MaxWeight: The maximal sum of weights on a given route between cell centres (limits the amount of "hills" that can be crossed during tracking)
  - WeightsBased: If multiple routes are possible, decides whether the optimal coice is based on distance or weight
  """
  InheritanceTree = []
  AnnotFramewiseSum = CalculateAnnotFramewiseSum(Annot)
  for path_key in AllPaths.keys():
    path = AllPaths[path_key]
    FirstFrame = -1
    for cell in path.values():
      if FirstFrame == -1 or cell.frame < FirstFrame:
        FirstFrame = cell.frame
    ci = CellInheritance(path[FirstFrame].origin, path)
    InheritanceTree.append(ci)
  for path_key in AllPaths.keys():
    path = AllPaths[path_key]
    FirstFrame = -1
    for cell in path.values():
      if FirstFrame == -1 or cell.frame < FirstFrame:
        FirstFrame = cell.frame
    if FirstFrame != 0:
      InputImg = Video[FirstFrame]
      InputImg = np.uint8(np.array(InputImg*(1-AnnotFramewiseSum[FirstFrame])))
      if DisplayMethodResults:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        ax1.imshow(InputImg, cmap=plt.cm.gray)
        ax2.imshow(AnnotationBinary[FirstFrame])
      AncestorCell = None
      MinLoss = -1
      MinRoute = None
      for path_origin in AllPaths.values():
        if FirstFrame in path_origin.keys() and path_origin[FirstFrame].id != path[FirstFrame].id:
          SquaredDist = np.power(path_origin[FirstFrame].x-path[FirstFrame].x,2)+np.power(path_origin[FirstFrame].y-path[FirstFrame].y,2)
          if SquaredDist<np.power(MaxInheritanceDistance,2):  
            Route, Weight = route_through_array(InputImg+BlackEpsilon, [path[FirstFrame].y, path[FirstFrame].x], 
                                                [path_origin[FirstFrame].y, path_origin[FirstFrame].x], fully_connected=True)
            Route = np.array(Route)
            Loss = None
            if WeightBased:
              Loss = Weight
            else:
              Loss = np.shape(Route)[0]
            if DisplayMethodResults:
              ax1.plot(Route[:, 1], Route[:, 0], '-r', lw=3)
            if Weight<MaxWeight and (MinLoss == -1 or Loss<MinLoss):
              MinLoss = Loss
              MinRoute = Route
              AncestorCell = path_origin[FirstFrame].origin
      if DisplayMethodResults and not MinRoute is None:
        ax1.plot(MinRoute[:, 1], MinRoute[:, 0], '-g', lw=3)
        plt.show()
        print("Loss: " + np.str(Loss))
      if AncestorCell != None:
        for CurrentCellPos in range(np.shape(InheritanceTree)[0]):
          if InheritanceTree[CurrentCellPos].origin.id == path[FirstFrame].origin.id:
            for ACell in InheritanceTree:
              if ACell.origin.id == AncestorCell.id:
                InheritanceTree[CurrentCellPos].add_parent(ACell)
          if InheritanceTree[CurrentCellPos].origin.id == AncestorCell.id:
            for CCell in InheritanceTree:
              if CCell.origin.id == path[FirstFrame].origin.id:
                InheritanceTree[CurrentCellPos].add_child(CCell)
  print("Inheritance tree successfully built")
  return InheritanceTree