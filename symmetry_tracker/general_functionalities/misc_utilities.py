import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools import mask as coco_mask
try:
  from IPython.display import HTML
except:
    pass

#Displayable progress bar
def progress(value, max=100):
  try:
    return HTML("""
      <progress
        value='{value}'
        max='{max}',
        style='width: 10%'
      >
        {value}
      </progress>
    """.format(value=value, max=max))
  except:
    print("IPython HTML progress bar unavailable in this environment")
    return None

#Function for calculation of center mass on an image using the moments of the image
#Used for calculating the center point of a cell
#There is a built in safety feature if Moment[00] would be 0, in that case it returns [None,None], signaling an error
def CenterMass(img):
  M = cv2.moments(np.array(img,dtype="uint8"))
  if M["m00"] !=0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
  else:
    return [None, None]
  return [cX,cY]

#Image rolling with custom paddig
#Source: https://stackoverflow.com/questions/2777907/python-numpy-roll-with-padding
def shift_2d_replace(data, dx, dy, constant=False):
  """
  Shifts the array in two dimensions while setting rolled values to constant
  :param data: The 2d numpy array to be shifted
  :param dx: The shift in x
  :param dy: The shift in y
  :param constant: The constant to replace rolled values with
  :return: The shifted array with "constant" where roll occurs
  """
  shifted_data = np.roll(data, dx, axis=1)
  if dx < 0:
    shifted_data[:, dx:] = constant
  elif dx > 0:
    shifted_data[:, 0:dx] = constant

  shifted_data = np.roll(shifted_data, dy, axis=0)
  if dy < 0:
    shifted_data[dy:, :] = constant
  elif dy > 0:
    shifted_data[0:dy, :] = constant
  return shifted_data

#Depth-first search on dictionary graph (recoursive method)
#Source: https://www.educative.io/edpresso/how-to-implement-depth-first-search-in-python 
def dfs(visited, graph, node):
  if node not in visited:
    visited.add(node)
    for neighbour in graph[node]:
        dfs(visited, graph, neighbour)

#Converts a figure to an image (numpy array)
#Source: https://web-backend.icare.univ-lille.fr/tutorials/convert_a_matplotlib_figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # Create a new canvas for the figure
    canvas = fig.canvas
    canvas.draw()

    # Get the width and height of the canvas
    w, h = canvas.get_width_height()

    # Get the RGBA buffer from the canvas
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    # Destroy the canvas explicitly to release resources
    plt.close(fig)

    return buf

#Calculates outer bounding box of multiple binary channels
def OuterBoundingBox(array):
    min_row, min_col = array.shape[1], array.shape[2]
    max_row, max_col = 0, 0
    for img in array:
        contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            min_row = min(min_row, y)
            min_col = min(min_col, x)
            max_row = max(max_row, y + h)
            max_col = max(max_col, x + w)
    return [min_row, min_col, max_row, max_col]

#Calculates the bounding box of a single binary array
def BoundingBox(array):
    contours, _ = cv2.findContours(array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_row, min_col = array.shape[0], array.shape[1]
    max_row, max_col = 0, 0
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        min_row, min_col = min(y, min_row), min(x, min_col)
        max_row, max_col = max(y + h, max_row), max(x + w, max_col)
    return [min_row, min_col, max_row, max_col]

#Calculates the overlapping area of two bounding boxes
def BoxOverlap(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    overlap_x = max(0, min(x2, x4) - max(x1, x3))
    overlap_y = max(0, min(y2, y4) - max(y1, y3))
    overlap_area = overlap_x * overlap_y
    return overlap_area

#Encodes a multi channel (C,D1,D2) array to RLE
def EncodeMultiRLE(Array):
  RLEList = []
  for Channel in Array:
    RLEList.append(coco_mask.encode(np.asfortranarray(Channel)))
  return RLEList

#Decodes a multi channel (C,D1,D2) array from RLE
def DecodeMultiRLE(RLEList):
  Array = []
  for Channel in RLEList:
    Array.append(coco_mask.decode(Channel))
  return np.array(Array, dtype = bool)