# The object containig a single frame occurence of a single cell
class CellInstance:
  def __init__(self, pos_x, pos_y, frame, id, annot, interpolated=False):
    self.x = pos_x
    self.y = pos_y
    self.id =  id
    self.frame = frame
    self.annot = annot
    self.origin = None
    self.interpolated = interpolated

  def print_data(self):
    print(self.x, end=" ")
    print(self.y, end="\t")
    print(self.id, end="\t")
    print(self.frame+1, end = '\t')
    if self.interpolated:
      print("Interpolated", end="\t")
    else:
      print("Normal cell", end="\t")
    print("")

  def define_origin(self, origin):
    self.origin = origin

# The object which contains how the cells got inherited, thus the whole inheritance tree can be built
# The "origin" field of the object is the first occurence of the given cell (this is a CellInstance type object)
class CellInheritance:
  def __init__(self, origin, full_path):
    self.origin = origin
    self.parents = []
    self.children = []
    self.full_path = full_path

  def add_parent(self, parent):
    self.parents.append(parent)

  def add_child(self, child):
    self.children.append(child)

  def print_data(self):
    print("Cell ID:", end =" ")
    print(self.origin.id, end ="\t")
    print("First frame:", end =" ")
    print(self.origin.frame+1, end ="\t")
    print("Parents:", end=" ")
    for p in self.parents:
      print(p.origin.id, end=" ")
    print("\tChildren:", end=" ")
    for c in self.children:
      print(c.origin.id, end=" ")
    print("")