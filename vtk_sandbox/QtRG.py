from PyQt4 import Qt, QtGui, QtCore
import RG 
from GetPot import GetPot
import math, re, vtk, global_vars
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor 
from Raven_GUI_Entity import *


#############################################################################
##  Qt module 
##
class QVtkInteractorWidget(QtGui.QFrame):

   def __init__(self, parent = None):
       super(QVtkInteractorWidget, self).__init__(parent)
       self.setup_ui()

   def setup_ui(self):
       self.setAttribute(QtCore.Qt.WA_ForceUpdatesDisabled)
       self.setAttribute(QtCore.Qt.WA_PaintOnScreen)
       self.setFocusPolicy(QtCore.Qt.StrongFocus)

       self.vbox = QtGui.QVBoxLayout(self)
       self.vbox.setMargin(0)
       self.interactor = QVTKRenderWindowInteractor(self)
       self.vbox.addWidget(self.interactor)
       self.setLayout(self.vbox)


class MainLayout(QtGui.QWidget):
    
    def __init__(self, cn):
        super(MainLayout, self).__init__()
	self.nodes = cn
        self.initUI()
        
    def initUI(self):      

        hbox = QtGui.QHBoxLayout(self)

        #topleft = QtGui.QFrame(self)
        topleft = QVtkInteractorWidget(self)
        topleft.setFrameShape(QtGui.QFrame.StyledPanel)
        cone = vtk.vtkConeSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cone.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
	actor.GetProperty().SetColor( 0, 1, 0 )
        renderer = vtk.vtkRenderer()

        render_window = topleft.interactor.GetRenderWindow()
        render_window.AddRenderer(renderer)
        renderer.AddActor(actor)
        render_window.Render()
 
        topright = QVtkInteractorWidget(self)
        topright.setFrameShape(QtGui.QFrame.StyledPanel)
        cone = vtk.vtkConeSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cone.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
	actor.GetProperty().SetColor( 0, 0, 1 )
        renderer = vtk.vtkRenderer()

        render_window = topright.interactor.GetRenderWindow()
        render_window.AddRenderer(renderer)
        renderer.AddActor(actor)
        render_window.Render()

        bottom = QtRG_Main(self, self.nodes)
   	bottom.callback_function = test_callback 

        splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter1.addWidget(topleft)
        splitter1.addWidget(topright)

        splitter2 = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitter2.addWidget(splitter1)
        splitter2.addWidget(bottom)

        hbox.addWidget(splitter2)
        self.setLayout(hbox)
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))
        
        self.show()
        
    def onChanged(self, text):
        self.lbl.setText(text)
        self.lbl.adjustSize()        
        

#############################################################################
##  Qt module 
##
#class QtRG_Main(QVTKRenderWindowInteractor):
class QtRG_Main(QVtkInteractorWidget):
   def __init__(self, parent, node):
      self.component_node = node
      self.callback_function = None
      #QVTKRenderWindowInteractor.__init__(self, None)
      super(QtRG_Main, self).__init__()

      # Create renderer--keep it because we'll need it to look up clicks
      self.renderer = vtk.vtkRenderer()
      #self.GetInteractor().GetRenderWindow().AddRenderer(self.renderer)

      render_window = self.interactor.GetRenderWindow()
      render_window.AddRenderer(self.renderer)
      render_window.Render()

      # Axes
      self.axes_actor = vtk.vtkAxesActor()
      self.axes_actor.DrawGridlinesOn = True
      self.axes_actor.TickVisibility = True
      self.axes_actor.AxisVisibility = False
      self.renderer.AddActor(self.axes_actor)
      global_vars.component_list = RG.CreateComponentsFromNode(node)
      
   	# Compute the geometry--this is a separate step because some stuff needs 
	   #   to be ready before rendering
      for c in global_vars.component_list:
         c.ComputeGeometry()
      
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
            self.renderer.AddActor(a)

   	# Put the camera on the positive X-Axis, with up being the positive Z-Axis
      self.renderer.GetActiveCamera().SetPosition(1.0, 0.0, 0.0);
      self.renderer.GetActiveCamera().SetViewUp(0.0, 0.0, 1.0);
      self.renderer.ResetCamera();
      self.renderer.SetBackground(.2, .3, .4);
      self.original_callback = self.interactor.mousePressEvent
      self.interactor.mousePressEvent = self.MPE 

   # Catch mouse clicks to allow component select
   #def mousePressEvent(self, ev):
   def MPE(self, ev):
      # Convert location to VTK Coords (3D)
      pos = (float(ev.pos().x()), 
               self.size().height() - float(ev.pos().y()), 0.0) 

      # Determine if there is an actor under this click
      pp = vtk.vtkPropPicker()
      pp.Pick(pos, self.renderer)
      selected_actor = pp.GetActor()

      # Don't do anything without a callback
      if self.callback_function != None:
         if selected_actor == None: 
            self.callback_function('None')
         else:
            # Correlate the actor to the component 
            for c in global_vars.component_list:
               al = c.ReportActors()
               if len(al) > 0 and selected_actor == al[0]:
                  self.callback_function(c.name)
                  return

      # Pass any unhandled events on to the base class so we can do the 
      #   panning, tilting, etc.
      self.original_callback(ev)


#############################################################################
##  An example callback from this module indicating that a particular 
##    entity has been selected in the GUI.
def test_callback(entity_name):
	print "QtRG.py: test_callback for entity: " + entity_name


#############################################################################
##  MAIN
##
if __name__ == "__main__":
   import sys, os
   app = QtGui.QApplication(sys.argv)

   # Read the input file name provided on the command line  
   if len(sys.argv) < 2:
      print sys.argv[0] + ": Usage: " + sys.argv[0] + " <input file name>"
   elif not os.path.exists(sys.argv[1]):
      print sys.argv[0] + ": Input file " + sys.argv[1] + " not found"
   else:
      print sys.argv[0] + ": Processing Input file " + sys.argv[1]
      # Load the Component Node 
      cn = RG.CreateComponentNodeFromFile(sys.argv[1])
      if cn == None:
         print sys.argv[0] + ": No component node found--exiting"
         sys.exit(1);

   # Create the display
   ml = MainLayout(cn)
   ml.setWindowTitle("Component Display for " + sys.argv[1])
   sys.exit(app.exec_())

