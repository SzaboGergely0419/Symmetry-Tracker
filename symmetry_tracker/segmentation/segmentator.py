import numpy as np
import cv2
import os
from pycocotools import mask as coco_mask
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

try:
  from IPython.display import display
  from symmetry_tracker.general_functionalities.misc_utilities import progress
except:
  pass

def PerformSegmentation(Predictor, VideoPath, Color = "GRAYSCALE", MinObjectSize = None):
  VideoFrames = sorted(os.listdir(VideoPath))
  Outmasks = {}

  NumFrames = len(VideoFrames)
  print("Performing segmentation")
  try:
    ProgressBar = display(progress(0, NumFrames), display_id=True)
  except:
    pass
  for Frame in range(0,NumFrames):
    try:
      ProgressBar.update(progress(Frame, NumFrames))
    except:
      pass
    if Color == "GRAYSCALE":
      Img = cv2.imread(os.path.join(VideoPath,VideoFrames[Frame]), cv2.IMREAD_GRAYSCALE)
      Img = np.expand_dims(Img, axis=2)
    elif Color == "RGB":
      Img = cv2.cvtColor(cv2.imread(os.path.join(VideoPath, VideoFrames[Frame])), cv2.COLOR_BGR2RGB)
    else:
      raise Exception(f"{Color} is an invalid keyword for Color")

    Outputs = Predictor(Img)
    Outmask = (Outputs["instances"].pred_masks.to("cpu").numpy())
    ReducedOutmask = []
    for Segment in Outmask:
      if MinObjectSize is None or np.sum(Segment) >= MinObjectSize:
        ReducedOutmask.append(coco_mask.encode(np.asfortranarray(Segment)))
    Outmasks[Frame] = ReducedOutmask
  try:
    ProgressBar.update(progress(1, 1))
  except:
    pass
  print("Segmentation finished")
  return Outmasks

def SingleVideoSegmentation(VideoPath, ModelPath, ModelConfigPath, Device, Color = "GRAYSCALE", ScoreThreshold = 0.2, MinObjectSize = None):
  """
  - VideoPath: The path to the video (in .png images format) to be segmented
  - ModelPath: The path to the model description file (.pth)
  - ModelConfigPath: The path to the model configuration file (.yaml)
  - Device: The device on which the segmentator should run (cpu or cuda:0)
  - Color: A keyword specific to the used model on the color encoding
           Available options: GRAYSCALE and RGB
  - ScoreThreshold: The acceptance threshold defining the object-ness of a given pixel
                    Lower values are more accepting
  - MinObjectSize: An optional parameter defining the minimal required object size in terms of pixel area
                  None is the default value
  """
  if not Color in ["GRAYSCALE", "RGB"]:
    raise Exception(f"{Color} is an invalid keyword for Color")

  cfg = get_cfg()
  cfg.merge_from_file(ModelConfigPath)
  cfg.MODEL.DEVICE = Device
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=ScoreThreshold
  cfg.MODEL.WEIGHTS = ModelPath
  Predictor = DefaultPredictor(cfg)
  Outmasks = PerformSegmentation(Predictor, VideoPath, Color, MinObjectSize)
  return Outmasks