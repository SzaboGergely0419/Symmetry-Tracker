import numpy as np
import cv2
import os
import mrc

def LoadVideoFromDV(VideoPath):
  Video = []
  VideoArray = mrc.imread(VideoPath)
  for i in range(VideoArray.shape[0]):
    img=(VideoArray[i,:,:,]/np.amax(VideoArray[i,:,:,]))*255.0
    Video.append(cv2.normalize(np.uint8(img), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
  print("Video loaded from "+VideoPath)
  return Video

def LoadVideoFromSingleTIFF(VideoPath, ChannelPeriodicity = None, StartPeriodicity = 2, MaxSampleNum=None):
  Video = []
  ret, TestImages = cv2.imreadmulti(VideoPath, [], cv2.IMREAD_ANYCOLOR)
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
    Video.append(cv2.normalize(np.uint8(TestImages[i]), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    SampleNum += 1
  print("Video loaded from "+VideoPath)
  return Video 

def LoadVideoFromImages(VideoPath, MaxSampleNum=None):
  Video = []
  ImageNames = os.listdir(VideoPath)
  ImageNames.sort()
  TestImages = []
  for ImageName in ImageNames:
    Image = cv2.imread(VideoPath+ImageName, cv2.IMREAD_GRAYSCALE)
    TestImages.append(Image)
  TestImages=np.asarray(TestImages)
  MinVal=np.amin(TestImages)
  TestImages-=MinVal
  MaxVal=np.amax(TestImages)
  SampleNum = 0
  for i in range(0,len(TestImages)):
    if MaxSampleNum!=None and SampleNum>=MaxSampleNum:
      break
    Video.append(cv2.normalize(np.uint8(TestImages[i]), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    SampleNum += 1
  print("Video loaded from "+VideoPath)
  return Video

def LoadVideo(VideoPath, VideoType, ChannelPeriodicity = None, StartPeriodicity = 2, MaxSampleNum=None):
  """
  - VideoPath: The path to a single video (DV, TIFF or multiple images in a folder)
  - VideoType: A keyword for the data type of the video
                Possible keywords: DV, TIFF, IMGs
  - ChannelPeriodicity: A parameter only for TIFF format files
                        Sets how frequently a non-fluorescent frame appeara
                        If None, all the frames are non-fluorescent channels
  - StartPeriodicity: Only important if ChannelPeriodicity is not None
                      Sets the first frame which is non-fluorescent
  - MaxSampleNum: A debugging parameter with which a limited number of frames can be loaded
                  Only usable with IMGs or TIFF 
  """
  Video = None
  if VideoType == "DV":
    Video = LoadVideoFromDV(VideoPath)
  elif VideoType == "TIFF":
    Video = LoadVideoFromSingleTIFF(VideoPath, ChannelPeriodicity, StartPeriodicity, MaxSampleNum)
  elif VideoType == "IMGs":
    Video = LoadVideoFromImages(VideoPath, MaxSampleNum)
  else:
    raise Exception(VideoType + " is not an appropriate keyword for VideoType")
  return Video