import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from skimage.measure import find_contours

from IPython.display import display
from general_functionalities.misc_utilities import progress

def PerformSegmentation(Predictor, Video, StartingFrame = None, EndingFrame = None, MinCellSize = None):
  Outmasks = {}
  if StartingFrame is None:
    StartingFrame = 0
  if EndingFrame is None:
    EndingFrame = np.shape(Video)[0]
  print("Performing segmentation")
  ProgressBar = display(progress(StartingFrame, EndingFrame), display_id=True)
  for Frame in range(StartingFrame,EndingFrame):
    ProgressBar.update(progress(Frame-StartingFrame, EndingFrame-StartingFrame))
    Img = np.expand_dims(Video[Frame], axis=2)
    Outputs = Predictor(Img)
    Outmask=(Outputs["instances"].pred_masks.to("cpu").numpy())
    ReducedOutmask = []
    if not(MinCellSize is None):
      for Segment in Outmask:
        if np.sum(Segment)>=MinCellSize:
          ReducedOutmask.append(Segment)
      ReducedOutmask = np.array(ReducedOutmask)
    else:
      ReducedOutmask = Outmask
    Outmasks[Frame]=ReducedOutmask
  ProgressBar.update(progress(1, 1))
  print("Segmentation finished")
  return Outmasks

def SingleVideoSegmentation(Video, ModelPath, ModelConfigPath, Device, ScoreThreshold = 0.2,
                            StartingFrame = None, EndingFrame = None, MinCellSize = None):
  """
  - Video: The video to be segmented
  - ModelPath: The path to the model description file (.pth)
  - ModelConfigPath: The path to the model configuration file (.yaml)
  - Device: The device on which the segmentator should run (cpu or cuda:0)
  - ScoreThreshold: The acceptance threshold defining the cell-ness of a given pixel
                    Lower values are more accepting
  - StartingFrame: The starting frame for the tracking.
                    The default None means minimal possible value
  - EndingFrame: The ending frame for the tracking. 
                  The default None means maximal possible value
  - MinCellSize: An optional parameter defining the minimal required cell size in terms of pixel area
                  None is the default value
  """
  cfg = get_cfg()
  cfg.merge_from_file(ModelConfigPath)
  cfg.MODEL.DEVICE = Device
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=ScoreThreshold
  cfg.MODEL.WEIGHTS = ModelPath
  Predictor = DefaultPredictor(cfg)
  Outmasks = PerformSegmentation(Predictor, Video, StartingFrame, EndingFrame, MinCellSize)
  return Outmasks