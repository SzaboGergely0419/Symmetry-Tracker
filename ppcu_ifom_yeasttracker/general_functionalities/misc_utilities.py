import numpy as np
import cv2
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
def fig2data ( fig ):
  """
  @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
  @param fig a matplotlib figure
  @return a numpy 3D array of RGBA values
  """
  # draw the renderer
  fig.canvas.draw ( )

  # Get the RGBA buffer from the figure
  w,h = fig.canvas.get_width_height()
  buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
  buf.shape = ( w, h,4 )

  # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
  buf = np.roll ( buf, 3, axis = 2 )
  return buf