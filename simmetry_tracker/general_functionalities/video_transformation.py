import numpy as np
import cv2
import os

def TransformVideoFromTIFF(InputPath, OutputPath, ChannelPeriodicity=1, StartPeriodicity=0, MaxSampleNum=None):
  """
  Converts a video in TIF format into a series of PNG images stored in a specified folder.

  Parameters:
  - InputPath (str): The path to the TIF-formatted video file.
  - OutputPath (str): The destination folder where the PNG images will be saved.
  - ChannelPeriodicity (int, optional): Determines how frequently a non-fluorescent frame appears.
                                        If set to None, all frames are non-fluorescent channels.
  - StartPeriodicity (int, optional): Only relevant if ChannelPeriodicity is not None.
                                      Specifies the first frame that is non-fluorescent.
  - MaxSampleNum (int, optional): A debugging parameter that limits the number of frames loaded.
  """

  ret, TestImages = cv2.imreadmulti(InputPath, [], cv2.IMREAD_ANYCOLOR)
  TestImages=np.asarray(TestImages)
  if not (ChannelPeriodicity is None):
    TestImages=TestImages[StartPeriodicity::ChannelPeriodicity,:,:]
  MinVal=np.amin(TestImages)
  TestImages-=MinVal
  MaxVal=np.amax(TestImages)
  SampleNum = 0
  for i in range(0,len(TestImages)):
    if MaxSampleNum!=None and SampleNum>=MaxSampleNum:
      break
    frame_path = os.path.join(OutputPath, f"frame_{i:04d}.png")
    cv2.imwrite(frame_path,cv2.normalize(np.uint8(TestImages[i]), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    SampleNum += 1
  print("Video transformed from "+InputPath)