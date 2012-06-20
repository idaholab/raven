from readInputFile import readInputFile, GPNode
from Raven_GUI_Entity import *
import vtk


#############################################################################
def GetAxesActor():
	axes = vtk.vtkAxes()

	# Adjust Axes Text Size
	ctp = axes.GetXAxisCaptionActor2D().GetCaptionTextProperty()
	ctp.SetWidth(0.20)
	ctp.SetHeight(0.05)
	ctp = axes.GetYAxisCaptionActor2D().GetCaptionTextProperty()
	ctp.SetWidth(0.20)
	ctp.SetHeight(0.05)
	ctp = axes.GetZAxisCaptionActor2D().GetCaptionTextProperty()
	ctp.SetWidth(0.10)
	ctp.SetHeight(0.025)

	axes_mapper = vtk.vtkPolyDataMapper()
	axes_tubes = vtk.vtkTubeFilter()
	axes_tubes.SetInputConnection(axes.GetOutputPort())
	axes_tubes.SetRadius(axes.GetScaleFactor() / 25.0)
	axes_tubes.SetNumberOfSides(6)
	axes_mapper.SetInputConnection(axes.GetOutputPort())
	axes_actor = vtk.vtkActor()
	axes_actor.SetMapper(axes_mapper)
	return axes_actor

#############################################################################
def CreateComponentNodeFromFile(file_name, print_messages = False):
 	rootNode = readInputFile(file_name)
	# Find the 'Components' node
	try:
		componentNode = rootNode.children['Components']
	except KeyError:
		if print_messages:
			print "CreateComponentsFromFile: Components node not found')"
		return None
		# raise ValueError('CreateComponentsFromFile: Components node not found')

	if print_messages:
		print "CreateComponentsFromFile: Components node found"
	return componentNode


#############################################################################
def CreateComponentsFromNode(componentNode, print_messages = False):
	component_count = 0
	#
	# Loop over the components and process them
	#
	rv = [ ]
	for c in componentNode.children:
		if print_messages:
			print "CreateComponentsFromFile: Processing component " + c
		# Get the next component node for processing
		new_obj = None
		comp = componentNode.children[c]

		# Get the component type
		t = GetParameter(comp, 'type')
		if t == None:
			if print_messages:
				print "CreateComponentsFromFile:   type not found...skipping"
		elif t == 'CoreChannel':
			new_obj = RGE_CoreChannel(comp)
		elif t == 'DummyTDV':
			new_obj = RGE_DummyTDV(comp)
		elif t == 'ErgBranch':
			new_obj = RGE_ErgBranch(comp)
		elif t == 'HeatExchanger':
			new_obj = RGE_HeatExchanger(comp)
		elif t == 'IdealPump':
			new_obj = RGE_IdealPump(comp)
		elif t == 'IsoBranch':
			new_obj = RGE_IsoBranch(comp)
		elif t == 'Pipe':
			new_obj = RGE_Pipe(comp)
		elif t == 'Pump':
			pass
		elif t == 'TDM':
			new_obj = RGE_TDM(comp)
		elif t == 'TimeDependentJunction':
			new_obj = RGE_TimeDependentJunction(comp)
		elif t == 'TimeDependentVolume':
			new_obj = RGE_TimeDependentVolume(comp)
		else:
			if print_messages:
				print "CreateComponentsFromFile:   Type " + t + \
						" not implemented--skipping"

		if new_obj != None:
			if print_messages:
				new_obj.Print()
			rv.append(new_obj)
			component_count += 1

	if print_messages:
		print "CreateComponentsFromFile: {0} component(s) found".format(component_count)
	return rv


#############################################################################
def CreateComponentsFromFile(file_name, print_messages = False):
	rv = [ ]
	# This will get the node to process...
	componentNode = CreateComponentNodeFromFile(file_name, print_messages)
	if componentNode != None:
		# This will convert the component node to a list of Raven_GUI_Entity-
		#   descended objects
		rv = CreateComponentsFromNode(componentNode, print_messages)

	return rv

#############################################################################
##  Handle mouse events
class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
	def __init__(self, r, i):
		self.renderer = r
		self.interactor = i
		self.outline_source = vtk.vtkOutlineSource()
		self.outline_actor = vtk.vtkActor()
		self.outline_tube_filter = vtk.vtkTubeFilter()
		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(self.outline_tube_filter.GetOutputPort())
		self.outline_actor.SetMapper(mapper)
		self.outline_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
		self.outline_actor.VisibilityOff()
		self.outline_tube_filter.SetRadius(0.01)
		self.outline_tube_filter.SetNumberOfSides(8)
		self.outline_tube_filter.SetInputConnection(self.outline_source.GetOutputPort())
		self.renderer.AddActor(self.outline_actor)
	
		self.corner_annotation = vtk.vtkCornerAnnotation()
		self.corner_annotation.SetLinearFontScaleFactor( 2 )
		self.corner_annotation.SetNonlinearFontScaleFactor( 1 )
		self.corner_annotation.SetMaximumFontSize( 20 )
		self.corner_annotation.GetTextProperty().SetColor(1.0, 1.0, 1.0)
		self.renderer.AddViewProp(self.corner_annotation);
		self.current_selected_actor = None
		self.user_callback_function = None

		self.SetDefaultRenderer(r)
		self.AddObserver('LeftButtonPressEvent', self.OnLeftButtonDown, 1.0) 

	def OnLeftButtonDown(self, obj, ev):
		import global_vars
		# Get the location of the click
		pos = self.interactor.GetEventPosition() + (0,)

		# Determine if there is an actor under this click
		pp = vtk.vtkPropPicker()
		pp.Pick(pos, global_vars.renderer)
		selected_actor = pp.GetActor()

		# If not, then clear the selection and get out
		if selected_actor == None:
			# Send event on to allow mouse movement of camera
			vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonDown(self)
			return;
		
		# OK, we picked something...find out what
		for c in global_vars.component_list:
			al = c.ReportActors()
			if len(al) > 0 and selected_actor == al[0]:
				if selected_actor == self.current_selected_actor:
					self.outline_actor.VisibilityOff()
					self.corner_annotation.SetText(0, "")
					self.current_selected_actor = None
				else:
					mapper = al[0].GetMapper()
					bounds = mapper.GetBounds()
					self.outline_source.SetBounds(bounds)
					self.outline_actor.VisibilityOn()
				
					out_text = c.name
					for pn in c.node.params:
						out_text += "\n" + pn + " = " 
						out_text += GetParameter(c.node, pn)

					self.corner_annotation.SetText(0, out_text)
					self.current_selected_actor = selected_actor

				# Update 
				self.renderer.GetRenderWindow().Render()
				if self.user_callback_function != None:
					self.user_callback_function(c.name)

	# A callback takes a string with the component name as the parameter
	def SetCallbackFunction(self, fn_name):
		self.user_callback_function = fn_name

#############################################################################
##  An example callback from this module indicating that a particular 
##    entity has been selected in the GUI.
def test_callback(entity_name):
	print "RG.py: test_callback for entity: " + entity_name


#############################################################################
if __name__ == '__main__':
	import sys, os
	import global_vars

	print '#################################################################'
	print '##  RAVEN GUI Starting Up...'
	print '#################################################################'

	# Read the input file name provided on the command line  
	if len(sys.argv) < 2:
		print sys.argv[0] + ": Usage: " + sys.argv[0] + " <input file name>"
	elif not os.path.exists(sys.argv[1]):
		print sys.argv[0] + ": Input file " + sys.argv[1] + " not found"
	else:
		print sys.argv[0] + ": Processing Input file " + sys.argv[1]
		global_vars.component_list = CreateComponentsFromFile(sys.argv[1])
		nc = len(global_vars.component_list)
		if nc == 0:
			sys.exit(1);
		
	print sys.argv[0] + ": Displaying {0} component(s)".format(nc)

	# Now we have a list of components...set up 
	import vtk

	#qinit = vtk.vtkQtInitialization()

	# create renderers (it's global so that 
	global_vars.renderer = vtk.vtkRenderer()
	#renderer.SetViewport(0, 0, 0.5, 1.0)

	# Compute the geometry--this is a separate step because some stuff needs 
	#   to be ready before rendering
	print sys.argv[0] + ": Computing Component Geometry"
	for c in global_vars.component_list:
		c.ComputeGeometry()

	print sys.argv[0] + ": Rendering Components"
	# Add on the actors from the components
	for c in global_vars.component_list:
		c.Render()
		c.RenderName()

	# Process any heat exchangers (which require the junctions to be computed) 
	for c in global_vars.component_list:
		if isinstance(c, RGE_HeatExchanger):
			c.CreateSecondary()

	# Draw the actors
	for c in global_vars.component_list:
		for a in c.ReportActors():
			global_vars.renderer.AddActor(a)

	# Put the camera on the positive X-Axis, with up being the positive Z-Axis
	global_vars.renderer.GetActiveCamera().SetPosition(1.0, 0.0, 0.0);
	global_vars.renderer.GetActiveCamera().SetViewUp(0.0, 0.0, 1.0);
	global_vars.renderer.ResetCamera();
	global_vars.renderer.SetBackground(.2, .3, .4);

	# Axes
	axes_actor = vtk.vtkAxesActor()
	axes_actor.DrawGridlinesOn = True
	axes_actor.TickVisibility = True
	axes_actor.AxisVisibility = False

	global_vars.renderer.AddActor(axes_actor)

	# Render It!
	renderWindow = vtk.vtkRenderWindow()
	# renderWindow.SetSize(600,300)
	renderWindow.AddRenderer(global_vars.renderer)
	iren = vtk.vtkRenderWindowInteractor()
	istyle = InteractorStyle(global_vars.renderer, iren)
	istyle.SetCallbackFunction(test_callback)
	iren.SetInteractorStyle(istyle)
	iren.SetRenderWindow(renderWindow)
	renderWindow.Render()
	renderWindow.SetWindowName("Component Display for " + sys.argv[1])
	iren.Start()

