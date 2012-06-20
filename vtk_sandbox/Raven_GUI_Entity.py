from GetPot import GetPot
import math, re, vtk, global_vars

VTK_LINE = 3

#############################################################################
def GetParameter(node, parameter_name):
	# Get the component type
	try:
		return node.params[parameter_name]
	except:
		return None

def StringToStringTuple(s):
	"""Return a tuple of space-separated string from a single string 
		Used to convert input parameters from GetPot nodes
		Note: Strings coming in will have quotes around them...remove them
	"""
	if not s:
		return None
	_s = s
	_s = _s.lstrip("'");
	_s = _s.rstrip("'");
	return tuple(_s.split(" "))

def StringToFloatTuple(s):
	"""Return a tuple of reals from a string containing space-separated values.
		Used to convert input file number sets such as "1 0 0"
		Note: Tuple strings coming in will have quotes around them
	"""
	if not s:
		return None
	return tuple([float(x) for x in StringToStringTuple(s)])


#############################################################################
class Raven_GUI_Entity(object):
	"""Represents one R7 system component.  All descend from here.
	"""
	def __init__(self, node):
		self.name = node.name
		self.node = node
		self.color = 1.0, 1.0, 1.0
		self.actor_list = [ ]
		# print 'Raven_GUI_Entity.__init__: Name = ' + self.name 

	def ComputeGeometry(self):
		pass

	def GetPortLocation(self, port_name):
		print 'Raven_GUI_Entity.GetPortLocation() Called!'
		return None

	def Print(self):
		print "Raven_GUI_Entity Properties for object " + self.name + ":"
		print "  Color = %s" % (self.color,)
		if self.node == None:
			print "  No Component Node Attached"
		else:
			print "  Component Node Values:"
			for pn in self.node.params:
				print "    " + pn + " = " + GetParameter(self.node, pn)

	def Render(self):
		print 'Raven_GUI_Entity.Render() Called!'
		raise NotImplementedError

	def RenderName(self):
		print 'Raven_GUI_Entity.RenderName() Called!'

	def ReportActors(self):
		return self.actor_list

	def RenderCaptionAt(self, location):
		ca = vtk.vtkCaptionActor2D()
		ca.SetCaption(self.name)
		ca.SetAttachmentPoint(location)
		ca.BorderOff()
		ca.GetCaptionTextProperty().BoldOff()
		ca.GetCaptionTextProperty().ItalicOff()
		ca.GetCaptionTextProperty().ShadowOff()
		ca.GetCaptionTextProperty().SetVerticalJustificationToCentered()
		ca.SetWidth(0.10)
		ca.SetHeight(0.025)
		ca.ThreeDimensionalLeaderOff()
		self.actor_list.append(ca)
		return

#############################################################################
class RGE_Pipe(Raven_GUI_Entity):
	"""Represents one pipe, including display of node values 
	"""
	def __init__(self, node):
		# Call the base class constructor
		Raven_GUI_Entity.__init__(self, node)
		# Load up the parameters needed for a pipe
		self.position = StringToFloatTuple(GetParameter(node, 'position'))
		self.orientation = StringToFloatTuple(GetParameter(node, 'orientation'))
		self.outlet_location = None
		# Make sure the orientation is a unit vector 
		#   (Must convert to list because item assignment is needed)
		l = list(self.orientation)
		vtk.vtkMath.Normalize(l);
		self.orientation = tuple(l)
		cross_section_area = float(GetParameter(node, 'A'))
		self.radius = math.sqrt(cross_section_area / math.pi)
		self.radius = 0.1
		self.length = float(GetParameter(node, 'length'))
		self.n_elems = int(GetParameter(node, 'n_elems'))

		# Create the VTK stuff
		self.poly_data = vtk.vtkPolyData()
		self.cells = vtk.vtkCellArray()
		self.tube_filter = vtk.vtkTubeFilter()
		self.tube_filter.SetInput(self.poly_data)
		self.tube_filter.SetNumberOfSides(10)
		self.tube_filter.SetRadius(self.radius)
		self.points = vtk.vtkPoints()
		self.scalars = vtk.vtkFloatArray()
		self.lut = vtk.vtkLookupTable()
		self.main_mapper = vtk.vtkPolyDataMapper()
		
	def ComputeGeometry(self):
		if self.n_elems < 1:
			self.n_elems = 1

		# Clear any previous stuff
		self.cells.Reset();
		self.points.Reset();
		self.scalars.Reset();
		self.poly_data.DeleteCells();

		# First point is position 
		self.points.InsertNextPoint(self.position);
		self.poly_data.GetCellData().SetScalars(self.scalars)

		# Divide the pipe into 'nElements' pieces.  The inlet is the starting 
		#   position of the pipe.
		step = tuple([ o * self.length / self.n_elems for o in self.orientation])
		vertex = self.position
		for idx in range(self.n_elems):
			vertex = tuple([ v + s for v, s in zip(vertex, step)])
			#self.points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
			self.points.InsertNextPoint(vertex)

		# The outlet location is also the last vertex...lets to the label 
		#   location (halfway between the inlet and outlet).
		self.outlet_location = vertex
		
		# We'll use the center later to show the label
		self.center_location = tuple((i + o) / 2.0 for \
					i, o in zip(self.position, self.outlet_location))

		# Now make linear cells between each pair successive pair of points
		# This will allow the display of cell data along the pipe
		self.poly_data.SetLines(self.cells)
		self.poly_data.SetPoints(self.points)
		ids = vtk.vtkIdList()
		for idx in range(self.n_elems):
			ids.Reset()
			ids.InsertNextId(idx)
			ids.InsertNextId(idx + 1)
			cell = self.poly_data.InsertNextCell(VTK_LINE, ids)
			self.scalars.InsertTuple1(cell, (idx + 0.5) / self.n_elems)			

	def Print(self):
		# Call the base class print
		Raven_GUI_Entity.Print(self)
		print "  RGE_Pipe Properties:"
		print "    Position = %s" % (self.position,)
		print "    Orientation =  %s " % (self.orientation,)
		print "    Radius = %f" % self.radius
		print "    Length = %f" % self.length
		print "    Number of Elements = %d" % self.n_elems

	def Render(self):
		self.actor_list = [ ]

		# Populate the color look-up table
		self.lut.SetNumberOfTableValues(100);
		self.lut.SetTableRange(0.0, 1.0);
		self.lut.SetHueRange(0.2, 0.8);
		self.lut.SetSaturationRange(0.2, 0.8);
		self.lut.SetAlphaRange(0.2, 0.8);
		self.lut.Build();

		# Prepare the actor(s) for display
		self.main_mapper.SetInputConnection(self.tube_filter.GetOutputPort())
		self.main_mapper.ScalarVisibilityOn()
		self.main_mapper.SetScalarModeToUseCellData()

		new_actor = vtk.vtkActor()
		new_actor.SetMapper(self.main_mapper)
		self.actor_list.append(new_actor)
		
	def RenderName(self):
		self.RenderCaptionAt(self.center_location)
		return
		import global_vars
		labelText = vtk.vtkVectorText()
		labelText.SetText(self.name)
		labelText.Update()
		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(labelText.GetOutputPort())
		follower = vtk.vtkFollower()
		follower.SetMapper(mapper)
		follower.SetScale(0.1, 0.1, 0.1);
		follower.RotateWXYZ(45, 0, 0, 1);
		center = follower.GetCenter()
		follower.AddPosition(self.radius * 1.5 - center[2], -center[0], -center[1]);
		follower.AddPosition(self.center_location)
		follower.SetMapper(mapper)
		follower.SetCamera(global_vars.renderer.GetActiveCamera())
		self.actor_list.append(follower)

	def GetPortLocation(self, port_name):
		#print "GetPortLocation(Pipe) " + self.name + " " + port_name
		# Return coordinates and radius
		if port_name == "in":
			return self.position + (self.radius,)
		elif port_name == "out":
			if self.outlet_location == None:
				return None
			return self.outlet_location + (self.radius,)
		else:
			return None

#############################################################################
class RGE_CoreChannel(RGE_Pipe):
	"""Represents one pipe, including display of node values 
	"""
	def __init__(self, node):
		# Call the parent class constructor
		RGE_Pipe.__init__(self, node)


#############################################################################
class RGE_HeatExchanger(RGE_Pipe):
	"""The heat exchanger is basically two thermally linked pipes.
	"""
	def __init__(self, node):
		# Call the parent class constructor
		RGE_Pipe.__init__(self, node)
		self.sec_inlet_location = None
		self.sec_outlet_location = None

	# Helper function to look up port information
	def LookupPortLocation(self, entity_name, port_name):
		#print "LookupPortLocation() " + entity_name + " " + port_name
		for c in global_vars.component_list:
			if c.name == entity_name:
				return c.GetPortLocation(port_name)
		# Didn't find one?  Return None
		return None

	#  Since coordinates are not given for the secondary inlet and outlet,
	#  we'll derive them from their connections...look them up from what 
	#  they are joined to.
	def CreateSecondary(self):
		inlet_name = self.name + '(secondary_in)'
		outlet_name = self.name + '(secondary_out)'
		# Iterate over the components to find the junctions connected to
		#   the exchanger secondary
		for c in global_vars.component_list:
			if isinstance(c, RGE_Junction):
				try:
					if c.inputs.find(outlet_name) != -1:
						if self.sec_outlet_location == None:
							self.sec_outlet_location = c.center; 
						else:
							print "Warning: Multiple secondary outlets found for " \
									+ c.name
					elif c.outputs.find(inlet_name) != -1:
						if self.sec_inlet_location == None:
							self.sec_inlet_location = c.center;
						else:
							print "Warning: Multiple secondary inlets found for " \
									+ c.name
				except:
					pass
		# Did we find them both?  If so, create the secondary
		if self.sec_outlet_location == None or self.sec_inlet_location == None:
			pass

		print self.sec_inlet_location
		print self.sec_outlet_location

	def GetPortLocation(self, port_name):
		# print "GetPortLocation(HX) " + self.name + " " + port_name
		# Return coordinates and radius
		if port_name == "primary_in":
			return self.position + (self.radius,)
		elif port_name == "primary_out":
			if self.outlet_location == None:
				return None
			return self.outlet_location + (self.radius,)
		else:
			return None

#############################################################################
class RGE_Junction(Raven_GUI_Entity):
	"""Represents a junction between several objects such as pipes 
	"""
	def __init__(self, node):
		# Call the base class constructor
		Raven_GUI_Entity.__init__(self, node)
		self.color = 0.2, 0.8, 0.2
		self.margin = 0.05
		self.num_connections = 0
		self.xMin = self.xMax = None
		self.yMin = self.yMax = None
		self.zMin = self.zMax = None
		self.center = None

		self.inputs = GetParameter(node, 'inputs')
		self.outputs = GetParameter(node, 'outputs') 

		# Create the VTK stuff
		self.cube_source = vtk.vtkCubeSource()
		self.main_mapper = vtk.vtkPolyDataMapper()

	# Helper function to look up port information
	def LookupPortLocation(self, entity_name, port_name):
		#print "LookupPortLocation() " + entity_name + " " + port_name
		for c in global_vars.component_list:
			if c.name == entity_name:
				return c.GetPortLocation(port_name)
		# Didn't find one?  Return None
		return None

	# Figure out where the connected joints are so we can draw a box around
	#   all of them. 
	def LocateJunctions(self):
		l = ()
		sst = StringToStringTuple(self.inputs)
		if sst != None:
			l = l + sst
		sst = StringToStringTuple(self.outputs)
		if sst != None:
			l = l + sst

		# Iterate through the inputs and outputs to figure out what size
		#   box is needed to show a joint
		is_first = True
		for p in l:
			# Get a set of numbers for this port--in form of 'entity(port)'
			# Note: This regex requires names to be alphanumeric only
			#print self.name + ": " + p
			m = re.search('^([\w-]{1,})(?:\()([\w-]{1,})(?:\))', p)	
			if m and len(m.groups()) == 2:
				loc = self.LookupPortLocation(m.groups()[0], m.groups()[1])
				if loc and len(loc) == 4:
					self.num_connections += 1
					if is_first:
						self.xMin = loc[0] - loc[3] - self.margin 
						self.xMax = loc[0] + loc[3] + self.margin 
						self.yMin = loc[1] - loc[3] - self.margin 
						self.yMax = loc[1] + loc[3] + self.margin 
						self.zMin = loc[2] - loc[3] - self.margin 
						self.zMax = loc[2] + loc[3] + self.margin 
						is_first = False
					else:
						if loc[0] - loc[3] - self.margin < self.xMin:
							self.xMin = loc[0] - loc[3] - self.margin
						if loc[0] + loc[3] + self.margin > self.xMax:
							self.xMax = loc[0] + loc[3] + self.margin
						if loc[1] - loc[3] - self.margin < self.yMin:
							self.yMin = loc[1] - loc[3] - self.margin
						if loc[1] + loc[3] + self.margin > self.yMax:
							self.yMax = loc[1] + loc[3] + self.margin
						if loc[2] - loc[3] - self.margin < self.zMin:
							self.zMin = loc[2] - loc[3] - self.margin
						if loc[2] + loc[3] + self.margin > self.zMax:
							self.zMax = loc[2] + loc[3] + self.margin

	def Print(self):
		# Call the base class print
		Raven_GUI_Entity.Print(self)
		print "  RGE_Junction Properties:"
		if self.inputs == None:
			print "    Inputs = None"
		else:
			print "    Inputs = %s" % (self.inputs,)
		if self.outputs == None:
			print "    Outputs = None"
		else:
			print "    Outputs = %s" % (self.outputs,)

	def Render(self):
		self.actor_list = [ ]

		self.LocateJunctions()

		if self.num_connections < 1:
			return

		self.cube_source.SetBounds(self.xMin, self.xMax, self.yMin, self.yMax, self.zMin, self.zMax)
		self.main_mapper.SetInputConnection(self.cube_source.GetOutputPort())

		new_actor = vtk.vtkActor()
		new_actor.GetProperty().SetColor(self.color)
		new_actor.SetMapper(self.main_mapper)
		self.actor_list.append(new_actor)

	def RenderName(self):
		try:
			self.center = ((self.xMax + self.xMin) / 2.0, 
						(self.yMax + self.yMin) / 2.0, 
						(self.zMax + self.zMin) / 2.0);
		except:
			print "Warning: Center not found for junction " + self.name
			self.Print()
			return
		# Uncomment this to use 2D Caption instead
		#self.RenderCaptionAt(center)
		#return

		# Fixed 3D text on side of junction (45 degree angle)
		label_text = vtk.vtkVectorText()
		label_text.SetText(self.name)
		label_text.Update()
		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(label_text.GetOutputPort())
		actor = vtk.vtkActor()
		actor.SetMapper(mapper)
		actor.SetScale(0.1, 0.1, 0.1);
		actor.RotateWXYZ(90, 0, 1, 0);
		actor.RotateWXYZ(135, 1, 0, 0);
		center = actor.GetCenter();
		actor.AddPosition(self.xMax + 0.1 - center[0], 
				(self.yMax + self.yMin) / 2.0 - center[1], 
				(self.zMax + self.zMin) / 2.0 - center[2]);
		actor.GetProperty().SetColor( 1, 1, 1 );
		self.actor_list.append(actor);

		# Text on opposite side of junction
		mapper2 = vtk.vtkPolyDataMapper()
		mapper2.SetInputConnection(label_text.GetOutputPort())
		actor2 = vtk.vtkActor()
		actor2.SetMapper(mapper)
		actor2.SetScale(0.1, 0.1, 0.1);
		actor2.RotateWXYZ(90, 0, 1, 0);
		actor2.RotateWXYZ(135, 1, 0, 0);
		actor2.RotateWXYZ(180, 0, 0, 1);
		center = actor2.GetCenter();
		actor2.AddPosition(self.xMin - 0.1 - center[0], 
				(self.yMax + self.yMin) / 2.0 - center[1], 
				(self.zMax + self.zMin) / 2.0 - center[2]);
		actor2.GetProperty().SetColor( 1, 1, 1 );
		self.actor_list.append(actor2);

#############################################################################
class RGE_ErgBranch(RGE_Junction):
	def __init__(self, node):
		# Call the base class constructor
		RGE_Junction.__init__(self, node)
		self.color = 0.2, 0.8, 0.2


#############################################################################
class RGE_TimeDependentJunction(RGE_Junction):
	def __init__(self, node):
		# Call the base class constructor
		RGE_Junction.__init__(self, node)
		self.color = 0.6, 0.6, 0.6
		self.inputs = GetParameter(node, 'input')


#############################################################################
class RGE_TimeDependentVolume(RGE_Junction):
	def __init__(self, node):
		# Call the base class constructor
		RGE_Junction.__init__(self, node)
		self.color = 0.0, 0.6, 0.6
		self.inputs = GetParameter(node, 'input')

#############################################################################
class RGE_IsoBranch(RGE_Junction):
	def __init__(self, node):
		# Call the base class constructor
		RGE_Junction.__init__(self, node)
		self.color = 0.7, 0.7, 0.7

#############################################################################
class RGE_TDM(RGE_TimeDependentJunction):
	def __init__(self, node):
		# Call the base class constructor
		RGE_TimeDependentJunction.__init__(self, node)
		self.color = 0.7, 0.0, 0.7

#############################################################################
class RGE_DummyTDV(RGE_TimeDependentJunction):
	def __init__(self, node):
		# Call the base class constructor
		RGE_TimeDependentJunction.__init__(self, node)

#############################################################################
class RGE_FlowJoint(RGE_Junction):
	def __init__(self, node):
		# Call the base class constructor
		RGE_Junction.__init__(self, node)
		self.color = 0.0, 0.4, 0.4


#############################################################################
class RGE_DummyPlenum(RGE_Junction):
	def __init__(self, node):
		# Call the base class constructor
		RGE_Junction.__init__(self, node)
		self.color = 0.4, 0.4, 0.0


#############################################################################
class RGE_IdealPump(RGE_Junction):
	def __init__(self, node):
		# Call the parent class constructor
		RGE_Junction.__init__(self, node)
		self.color = 0.7, 0.7, 0.0
