'''
Created on Nov 14, 2013

@author: alfoa
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

import numpy as np
from BaseType import BaseType
import copy
import ast
from scipy.interpolate import *
import importlib
from math import pow  
from math import sqrt  

def removeNanEntries(X):
  return X[~np.isnan(X).any(1)]

class OutStreamManager(BaseType):
  '''
  ********************************************************************
  *                          OUTSTREAM CLASS                         *
  ********************************************************************
  *  This class is a general base class for outstream action classes *
  *  For example, a matplotlib interface class or Print class, etc.  *
  ********************************************************************
  '''
  def __init__(self):
    '''
      Init of Base class 
    '''
    BaseType.__init__(self)
    # outstreaming options
    self.options = {}
    # we are in interactive mode?
    self.interactive = True
    #counter
    self.counter = 0
    #overwrite outstream?
    self.overwrite = True
    # outstream types available
    self.availableOutStreamType = []
    # number of agregated outstreams
    self.numberAggregatedOS = 1

  def readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the got inputs 
    @ In, xmlNode    : Xml element node
    @ Out, None
    '''
    BaseType.readMoreXML(self,xmlNode)
    if 'interactive' in xmlNode.attrib.keys():
      if xmlNode.attrib['interactive'].lower() in ['t','true','on']: self.interactive = True
      else: self.interactive = False
    if 'overwrite' in xmlNode.attrib.keys():
      if xmlNode.attrib['overwrite'].lower() in ['t','true','on']: self.overwrite = True
      else: self.overwrite = False
    self.localReadXML(xmlNode)  

  def addInitParams(self,tempDict):
    '''
    Function adds the initial parameter in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    '''
    tempDict[                     'Global Class Type                 '] = 'OutStreamManager'
    tempDict[                     'Specialized Class Type            '] = self.type
    if self.interactive: tempDict['Interactive mode                  '] = 'True'
    else:                tempDict['Interactive mode                  '] = 'False'
    if self.overwrite:   tempDict['Overwrite output everytime called '] = 'True'
    else:                tempDict['Overwrite output everytime called '] = 'False'
    for index in range(len((self.availableOutStreamType))) : tempDict['OutStream Available #'+str(index+1)+'   :'] = self.availableOutStreamType[index]
    self.localAddInitParams(tempDict)
    return tempDict

  def addOutput(self):
    '''
    Function to add a new output source (for example a CSV file or a HDF5 object)
    @ In, toLoadFrom, source object
    @ Out, None 
    '''
    raise NotImplementedError('STREAM MANAGER: ERROR -> method addOutput must be implemented by derived classes!!!!')

  def initialize(self,inDict):
    '''
    Function to initialize the OutStream. It basically looks for the "data" object and link it to the system
    @ In, inDict, dictionary, It contains all the Object are going to be used in the current step. The sources are searched into this.
    @ Out, None 
    '''
    self.sourceData   = []
    for agrosindex in range(self.numberAggregatedOS):
      foundData = False
      for output in inDict['Output']:
        if output.name.strip() == self.sourceName[agrosindex]:
          self.sourceData.append(output)
          foundData = True
      if not foundData:
        for inp in inDict['Input']:
          if not isinstance(inp, basestring):
            if inp.name.strip() == self.sourceName[agrosindex]:
              self.sourceData.append(inp)
              foundData = True  
      if not foundData and 'TargetEvaluation' in inDict.keys():
        if inDict['TargetEvaluation'].name.strip() == self.sourceName[agrosindex]:
          self.sourceData.append(inDict['TargetEvaluation'])
          foundData = True 
      if not foundData and 'SolutionExport' in inDict.keys():
        if inDict['SolutionExport'].name.strip() == self.sourceName[agrosindex]:
          self.sourceData.append(inDict['SolutionExport'])
          foundData = True 
      if not foundData: raise IOError('STREAM MANAGER: ERROR -> the Data named ' + self.sourceName[agrosindex] + ' has not been found!!!!')
#
#
#
class OutStreamPlot(OutStreamManager):
  def __init__(self):
    OutStreamManager.__init__(self)
    self.type         = 'OutStreamPlot'
    # available 2D and 3D plot types
    self.availableOutStreamTypes = {2:['scatter','line','histogram','stem','step','polar','pseudocolor'], 
                                    3:['scatter','line','stem','surface','wireframe','tri-surface',
                                       'contour','filled_contour','contour3D','filled_contour3D','histogram']}
    # default plot is 2D
    self.dim          = 2
    # list of source names
    self.sourceName   = []
    # source of data
    self.sourceData   = None
    # dictionary of x,y,z coordinates
    self.x_cordinates = None
    self.y_cordinates = None
    self.z_cordinates = None
    # dictionary of x,y,z values
    self.x_values = None
    self.y_values = None
    self.z_values = None
    # color map
    self.color_map_coordinates = None
    self.color_map_values      = None
    # list of the outstream types
    self.outStreamTypes = []
    # interpolate functions available
    self.interpAvail = ['nearest','linear','cubic','multiquadric','inverse','gaussian','Rbflinear','Rbfcubic','quintic','thin_plate']
    
  #####################
  #  PRIVATE METHODS  #
  #####################

  def __splitVariableNames(self,what,where):
    ''' 
      Function to split the variable names
      @ In, what => x,y,z or color_map
      @ In, where, tuple => pos 0 = plotIndex, pos 1 = variable Index 
    '''
    if   what == 'x'        : var = self.x_cordinates[where[0]][where[1]]
    elif what == 'y'        : var = self.y_cordinates[where[0]][where[1]]
    elif what == 'z'        : var = self.z_cordinates[where[0]][where[1]]
    elif what == 'color_map': var = self.color_map_coordinates[where[0]][where[1]]
    # the variable can contain brackets (when the symbol "|" is present in the variable name), 
    # for example DataName|Input|(RavenAuxiliary|variableName|initial_value)
    # or it can look like DataName|Input|variableName
    if var:
      if '(' in var and ')' in var:
        if var.count('(') > 1: raise IOError('STREAM MANAGER: ERROR -> In Plot ' +self.name +'.Only a couple of () is allowed in variable names!!!!!!')
        result = var.split('|(')[0].split('|')
        result.append(var.split('(')[1].replace(")", ""))
      else:  result = var.split('|')
    else: result = None
    if len(result) != 3: raise IOError('STREAM MANAGER: ERROR -> In Plot ' +self.name +'.Only three level variables are accepted !!!!!')
    return result
      
  def __readPlotActions(self,snode):
    ''' 
      Function to read, from the xml input, the actions that are required to be performed on the Plot 
      @ In, snode => xml node
    '''
    for node in snode:
      self.options[node.tag] = {}
      if len(node):
        for subnode in node: 
          if subnode.tag != 'kwargs': 
            self.options[node.tag][subnode.tag] = subnode.text
            if not subnode.text: raise IOError('STREAM MANAGER: ERROR -> In Plot ' +self.name +'. Problem in sub-tag ' + subnode.tag + ' in '+node.tag+' block. Please check!')
          else:
            self.options[node.tag]['attributes'] = {} 
            for subsub in subnode: 
              self.options[node.tag]['attributes'][subsub.tag] = subsub.text
              if not subnode.text: raise IOError('STREAM MANAGER: ERROR -> In Plot ' +self.name +'. Problem in sub-tag ' + subnode.tag + ' in '+node.tag+' block. Please check!')   
      elif node.text: 
        if node.text.strip(): self.options[node.tag][node.tag] = node.text
    if 'how' not in self.options.keys(): self.options['how']={'how':'screen'} 

  def __fillCoordinatesFromSource(self):
    ''' 
      Function to retrieve the pointers of the data values (x,y,z) 
      @ In, None
      @ Out, boolean, true if the data are filled, false otherwise
    '''
    self.x_values = []
    if self.y_cordinates: self.y_values = []
    if self.z_cordinates: self.z_values = []
    if self.color_map_coordinates: self.color_map_values = []
    for pltindex in range(len(self.outStreamTypes)):
      self.x_values.append(None)
      if self.y_cordinates: self.y_values.append(None)
      if self.z_cordinates: self.z_values.append(None)
      if self.color_map_coordinates: self.color_map_values.append(None)
    for pltindex in range(len(self.outStreamTypes)):
      if self.sourceData[pltindex].isItEmpty(): return False
      if self.sourceData[pltindex].type.strip() not in 'Histories': 
        self.x_values[pltindex] = {1:[]}
        if self.y_cordinates: self.y_values[pltindex] = {1:[]}
        if self.z_cordinates: self.z_values[pltindex] = {1:[]}
        if self.color_map_coordinates: self.color_map_values[pltindex] = {1:[]}
        for i in range(len(self.x_cordinates[pltindex])):
          xsplit = self.__splitVariableNames('x', (pltindex,i)) 
          parame = self.sourceData[pltindex].getParam(xsplit[1],xsplit[2])
          if type(parame) == np.ndarray: self.x_values[pltindex][1].append(np.asarray(parame))
          else:
            conarr = np.zeros(len(parame.keys()))
            index = 0
            for item in parame.values(): conarr[index] = item[0]; index += 1
            self.x_values[pltindex][1].append(np.asarray(conarr))           
        if self.y_cordinates:
          for i in range(len(self.y_cordinates[pltindex])): 
            ysplit = self.__splitVariableNames('y', (pltindex,i))
            parame = self.sourceData[pltindex].getParam(ysplit[1],ysplit[2])
            if type(parame) == np.ndarray: self.y_values[pltindex][1].append(np.asarray(parame))
            else:
              conarr = np.zeros(len(parame.keys())) 
              index = 0
              for item in parame.values(): conarr[index] = item[0]; index += 1
              self.y_values[pltindex][1].append(np.asarray(conarr))           
        if self.z_cordinates and self.dim>2:
          for i in range(len(self.z_cordinates[pltindex])):
            zsplit = self.__splitVariableNames('z', (pltindex,i)) 
            parame = self.sourceData[pltindex].getParam(zsplit[1],zsplit[2])
            if type(parame) == np.ndarray: self.z_values[pltindex][1].append(np.asarray(parame))
            else:
              conarr = np.zeros(len(parame.keys())) 
              for index in range(len(parame.values())): conarr[index] = parame.values()[index][0]
              self.z_values[pltindex][1].append(np.asarray(conarr))  
        if self.color_map_coordinates:
          for i in range(len(self.color_map_coordinates[pltindex])):
            zsplit = self.__splitVariableNames('color_map', (pltindex,i)) 
            parame = self.sourceData[pltindex].getParam(zsplit[1],zsplit[2])
            if type(parame) == np.ndarray: self.color_map_values[pltindex][1].append(np.asarray(parame))
            else:
              conarr = np.zeros(len(parame.keys())) 
              for index in range(len(parame.values())): conarr[index] = parame.values()[index][0]
              self.color_map_values[pltindex][1].append(np.asarray(conarr))  
      else:
        #Histories
        self.x_values[pltindex] = {}
        if self.y_cordinates: self.y_values[pltindex] = {}
        if self.z_cordinates  and self.dim>2: self.z_values[pltindex] = {}
        if self.color_map_coordinates: self.color_map_values[pltindex] = {}
        for cnt,key in enumerate(self.sourceData[pltindex].getInpParametersValues().keys()): 
          #the key is the actual history number (ie 1, 2 , 3 etc)
          self.x_values[pltindex][cnt] = []
          if self.y_cordinates: self.y_values[pltindex][cnt] = []
          if self.z_cordinates: self.z_values[pltindex][cnt] = []
          if self.color_map_coordinates: self.color_map_values[pltindex][cnt] = []
          for i in range(len(self.x_cordinates[pltindex])): 
            xsplit = self.__splitVariableNames('x', (pltindex,i)) 
            self.x_values[pltindex][cnt].append(np.asarray(self.sourceData[pltindex].getParam(xsplit[1],cnt+1)[xsplit[2]]))
          if self.y_cordinates:
            for i in range(len(self.y_cordinates[pltindex])): 
              ysplit = self.__splitVariableNames('y', (pltindex,i))
              self.y_values[pltindex][cnt].append(np.asarray(self.sourceData[pltindex].getParam(ysplit[1],cnt+1)[ysplit[2]]))
          if self.z_cordinates and self.dim>2:
            for i in range(len(self.z_cordinates[pltindex])): 
              zsplit = self.__splitVariableNames('z', (pltindex,i))
              self.z_values[pltindex][cnt].append(np.asarray(self.sourceData[pltindex].getParam(zsplit[1],cnt+1)[zsplit[2]]))
          if self.color_map_coordinates:
            for i in range(len(self.color_map_coordinates[pltindex])): 
              zsplit = self.__splitVariableNames('color_map', (pltindex,i))
              self.color_map_values[pltindex][cnt].append(np.asarray(self.sourceData[pltindex].getParam(zsplit[1],cnt+1)[zsplit[2]]))
      #check if something has been got or not
      if len(self.x_values[pltindex].keys()) == 0: return False
      else:
        for key in self.x_values[pltindex].keys():
          if len(self.x_values[pltindex][key]) == 0: return False
          else:
            for i in range(len(self.x_values[pltindex][key])):
              if self.x_values[pltindex][key][i].size == 0: return False 
      if self.y_cordinates:
        if len(self.y_values[pltindex].keys()) == 0: return False    
        else:
          for key in self.y_values[pltindex].keys():
            if len(self.y_values[pltindex][key]) == 0: return False    
            else:
              for i in range(len(self.y_values[pltindex][key])):
                if self.y_values[pltindex][key][i].size == 0: return False        
      if self.z_cordinates and self.dim>2:
        if len(self.z_values[pltindex].keys()) == 0: return False
        else:
          for key in self.z_values[pltindex].keys():
            if len(self.z_values[pltindex][key]) == 0: return False      
            else:
              for i in range(len(self.z_values[pltindex][key])):
                if self.z_values[pltindex][key][i].size == 0: return False    
      if self.color_map_coordinates:
        if len(self.color_map_values[pltindex].keys()) == 0: return False
        else:
          for key in self.color_map_values[pltindex].keys():
            if len(self.color_map_values[pltindex][key]) == 0: return False      
            else:
              for i in range(len(self.color_map_values[pltindex][key])):
                if self.color_map_values[pltindex][key][i].size == 0: return False      
    return True  
  
  def __executeActions(self):
    ''' 
      Function to execute the actions must be performed on the Plot(for example, set the x,y,z axis ranges, etc)
      @ In, None
    '''
    if 'label_format' not in self.options.keys(): 
      if self.dim == 2: self.plt.ticklabel_format(**{'style':'sci','scilimits':(0,0),'useOffset':False,'axis':'both'})
      if self.dim == 3: self.plt3D.ticklabel_format(**{'style':'sci','scilimits':(0,0),'useOffset':False,'axis':'both'})
    if 'title'        not in self.options.keys():
      if self.dim == 2: self.plt.title(self.name,fontdict={'verticalalignment':'baseline','horizontalalignment':'center'})
      if self.dim == 3: self.plt3D.set_title(self.name,fontdict={'verticalalignment':'baseline','horizontalalignment':'center'})    
    for key in self.options.keys():
      if   key in ['how','plot_settings','figure_properties']: pass
      elif key == 'range': 
        if self.dim == 2:
          if 'ymin' in self.options[key].keys(): self.plt.ylim(ymin = ast.literal_eval(self.options[key]['ymin']))
          if 'ymax' in self.options[key].keys(): self.plt.ylim(ymax = ast.literal_eval(self.options[key]['ymax']))
          if 'xmin' in self.options[key].keys(): self.plt.xlim(xmin = ast.literal_eval(self.options[key]['xmin']))
          if 'xmax' in self.options[key].keys(): self.plt.xlim(xmax = ast.literal_eval(self.options[key]['xmax']))
        elif self.dim == 3:
          if 'xmin' in self.options[key].keys(): self.plt3D.set_xlim3d(xmin = ast.literal_eval(self.options[key]['xmin']))
          if 'xmax' in self.options[key].keys(): self.plt3D.set_xlim3d(xmax = ast.literal_eval(self.options[key]['xmax']))
          if 'ymin' in self.options[key].keys(): self.plt3D.set_ylim3d(ymin = ast.literal_eval(self.options[key]['ymin']))
          if 'ymax' in self.options[key].keys(): self.plt3D.set_ylim3d(ymax = ast.literal_eval(self.options[key]['ymax']))
          if 'zmin' in self.options[key].keys(): self.plt3D.set_zlim(ast.literal_eval(self.options[key]['zmin']),ast.literal_eval(self.options[key]['zmax']))      
      elif key == 'label_format':
        if 'style' not in self.options[key].keys(): self.options[key]['style'        ]   = 'sci'
        if 'limits' not in self.options[key].keys(): self.options[key]['limits'      ] = '(0,0)'
        if 'useOffset' not in self.options[key].keys(): self.options[key]['useOffset'] = 'False'
        if 'axis' not in self.options[key].keys(): self.options[key]['axis'          ] = 'both'
        if self.dim == 2:  self.plt.ticklabel_format(**{'style':self.options[key]['style'],'scilimits':ast.literal_eval(self.options[key]['limits']),'useOffset':ast.literal_eval(self.options[key]['useOffset']),'axis':self.options[key]['axis']})          
        elif self.dim == 3:self.plt3D.ticklabel_format(**{'style':self.options[key]['style'],'scilimits':ast.literal_eval(self.options[key]['limits']),'useOffset':ast.literal_eval(self.options[key]['useOffset']),'axis':self.options[key]['axis']})        
      elif key == 'camera': 
        if self.dim == 2: print('STREAM MANAGER: Warning -> 2D plots have not a camera attribute... They are 2D!!!!')
        elif self.dim == 3:
          if 'elevation' in self.options[key].keys() and 'azimuth' in self.options[key].keys():       self.plt3D.view_init(elev = float(self.options[key]['elevation']),azim = float(self.options[key]['azimuth']))
          elif 'elevation' in self.options[key].keys() and 'azimuth' not in self.options[key].keys(): self.plt3D.view_init(elev = float(self.options[key]['elevation']),azim = None)
          elif 'elevation' not in self.options[key].keys() and 'azimuth' in self.options[key].keys(): self.plt3D.view_init(elev = None,azim = float(self.options[key]['azimuth']))
      elif key == 'title':
        if self.dim == 2:
          if 'attributes' in self.options[key].keys(): self.plt.title(self.options[key]['text'],**self.options[key]['attributes'])
          else:                                        self.plt.title(self.options[key]['text'])            
        elif self.dim == 3:
          if 'attributes' in self.options[key].keys(): self.plt3D.set_title(self.options[key]['text'],**self.options[key]['attributes'])
          else: self.plt3D.set_title(self.options[key]['text'])  
      elif key== 'scale':
        if self.dim == 2:
          if 'xscale' in self.options[key].keys(): self.plt.xscale(self.options[key]['xscale'])
          if 'yscale' in self.options[key].keys(): self.plt.yscale(self.options[key]['yscale'])
        elif self.dim == 3:
          if 'xscale' in self.options[key].keys(): self.plt3D.set_xscale(self.options[key]['xscale'])
          if 'yscale' in self.options[key].keys(): self.plt3D.set_yscale(self.options[key]['yscale'])        
          if 'zscale' in self.options[key].keys(): self.plt3D.set_zscale(self.options[key]['zscale'])     
      elif key == 'add_text':
        if 'position' not in self.options[key].keys(): self.options[key]['position'] = str((min(self.x_values) + max(self.x_values))*0.5) + ',' + str((min(self.y_values) + max(self.y_values))*0.5)
        if 'withdash' not in self.options[key].keys(): self.options[key]['withdash'] = 'False' 
        if 'fontdict' not in self.options[key].keys(): self.options[key]['fontdict'] = 'None'
        else: 
          try: self.options[key]['fontdict'] = ast.literal_eval(self.options[key]['fontdict'])
          except AttributeError: raise('STREAM MANAGER: ERROR -> In ' + key +' tag: can not convert the string "' + self.options[key]['fontdict'] + '" to a dictionary! Check syntax for python function ast.literal_eval')
        if self.dim == 2 :
          if 'attributes' in self.options[key].keys(): self.plt.text(float(self.options[key]['position'].split(',')[0]),float(self.options[key]['position'].split(',')[1]),self.options[key]['text'],fontdict=self.options[key]['fontdict'],**self.options[key]['attributes'])
          else: self.plt.text(ast.literal_eval(self.options[key]['position'].split(',')[0]),ast.literal_eval(self.options[key]['position'].split(',')[1]),self.options[key]['text'],fontdict=self.options[key]['fontdict']) 
        elif self.dim ==3:
          if 'attributes' in self.options[key].keys(): self.plt3D.text(float(self.options[key]['position'].split(',')[0]),float(self.options[key]['position'].split(',')[1]),float(self.options[key]['position'].split(',')[2]),self.options[key]['text'],fontdict=ast.literal_eval(self.options[key]['fontdict']),withdash=ast.literal_eval(self.options[key]['withdash']),**self.options[key]['attributes'])
          else: self.plt3D.text(float(self.options[key]['position'].split(',')[0]),float(self.options[key]['position'].split(',')[1]),float(self.options[key]['position'].split(',')[2]),self.options[key]['text'],fontdict=ast.literal_eval(self.options[key]['fontdict']),withdash=ast.literal_eval(self.options[key]['withdash']))
      elif key == 'autoscale':
          if 'enable' not in self.options[key].keys(): self.options[key]['enable'] = 'True'
          elif self.options[key]['enable'].lower() in ['t','true']: self.options[key]['enable'] = 'True'
          elif self.options[key]['enable'].lower() in ['f','false']: self.options[key]['enable'] = 'False' 
          if 'axis' not in self.options[key].keys()  : self.options[key]['axis'] = 'both'
          if 'tight' not in self.options[key].keys() : self.options[key]['tight'] = 'None'        
          if self.dim == 2  : self.plt.autoscale(enable = ast.literal_eval(self.options[key]['enable']), axis = self.options[key]['axis'], tight = ast.literal_eval(self.options[key]['tight']))
          elif self.dim == 3: self.plt3D.autoscale(enable = ast.literal_eval(self.options[key]['enable']), axis = self.options[key]['axis'], tight = ast.literal_eval(self.options[key]['tight']))
      elif key == 'horizontal_line':
        if self.dim == 3: print('STREAM MANAGER: Warning -> horizontal_line not available in 3-D plots!!')
        elif self.dim == 2:
          if 'y' not in self.options[key].keys(): self.options[key]['y'] = '0'
          if 'xmin' not in self.options[key].keys()  : self.options[key]['xmin'] = '0'
          if 'xmax' not in self.options[key].keys() : self.options[key]['xmax'] = '1'
          if 'hold' not in self.options[key].keys() : self.options[key]['hold'] = 'None'
          if 'attributes' in self.options[key].keys(): self.plt.axhline(y=ast.literal_eval(self.options[key]['y']), xmin=ast.literal_eval(self.options[key]['xmin']), xmax=ast.literal_eval(self.options[key]['xmax']), hold=ast.literal_eval(self.options[key]['hold']),**self.options[key]['attributes'])
          else: self.plt.axhline(y=ast.literal_eval(self.options[key]['y']), xmin=ast.literal_eval(self.options[key]['xmin']), xmax=ast.literal_eval(self.options[key]['xmax']), hold=ast.literal_eval(self.options[key]['hold']))
      elif key == 'vertical_line':
        if self.dim == 3: print('STREAM MANAGER: Warning -> vertical_line not available in 3-D plots!!')
        elif self.dim == 2:
          if 'x' not in self.options[key].keys(): self.options[key]['x'] = '0'
          if 'ymin' not in self.options[key].keys()  : self.options[key]['ymin'] = '0'
          if 'ymax' not in self.options[key].keys() : self.options[key]['ymax'] = '1'
          if 'hold' not in self.options[key].keys() : self.options[key]['hold'] = 'None'
          if 'attributes' in self.options[key].keys(): self.plt.axhline(x=ast.literal_eval(self.options[key]['x']), ymin=ast.literal_eval(self.options[key]['ymin']), ymax=ast.literal_eval(self.options[key]['ymax']), hold=ast.literal_eval(self.options[key]['hold']),**self.options[key]['attributes'])
          else: self.plt.axvline(x=ast.literal_eval(self.options[key]['x']), ymin=ast.literal_eval(self.options[key]['ymin']), ymax=ast.literal_eval(self.options[key]['ymax']), hold=ast.literal_eval(self.options[key]['hold']))
      elif key == 'horizontal_rectangle':
        if self.dim == 3: print('STREAM MANAGER: Warning -> horizontal_rectangle not available in 3-D plots!!')
        elif self.dim == 2:
          if 'ymin' not in self.options[key].keys(): raise('STREAM MANAGER: ERROR -> ymin parameter is needed for function horizontal_rectangle!!')
          if 'ymax' not in self.options[key].keys(): raise('STREAM MANAGER: ERROR -> ymax parameter is needed for function horizontal_rectangle!!')
          if 'xmin' not in self.options[key].keys()  : self.options[key]['xmin'] = '0'
          if 'xmax' not in self.options[key].keys() : self.options[key]['xmax'] = '1'
          if 'attributes' in self.options[key].keys(): self.plt.axhspan(ast.literal_eval(self.options[key]['ymin']),ast.literal_eval(self.options[key]['ymax']), ymin=ast.literal_eval(self.options[key]['xmin']), ymax=ast.literal_eval(self.options[key]['xmax']),**self.options[key]['attributes'])
          else:self.plt.axhspan(ast.literal_eval(self.options[key]['ymin']),ast.literal_eval(self.options[key]['ymax']), xmin=ast.literal_eval(self.options[key]['xmin']), xmax=ast.literal_eval(self.options[key]['xmax']))
      elif key == 'vertical_rectangle':
        if self.dim == 3: print('STREAM MANAGER: Warning -> vertical_rectangle not available in 3-D plots!!')
        elif self.dim == 2:
          if 'xmin' not in self.options[key].keys(): raise('STREAM MANAGER: ERROR -> xmin parameter is needed for function vertical_rectangle!!')
          if 'xmax' not in self.options[key].keys(): raise('STREAM MANAGER: ERROR -> xmax parameter is needed for function vertical_rectangle!!')
          if 'ymin' not in self.options[key].keys()  : self.options[key]['ymin'] = '0'
          if 'ymax' not in self.options[key].keys() : self.options[key]['ymax'] = '1'
          if 'attributes' in self.options[key].keys(): self.plt.axvspan(ast.literal_eval(self.options[key]['xmin']),ast.literal_eval(self.options[key]['xmax']), ymin=ast.literal_eval(self.options[key]['ymin']), ymax=ast.literal_eval(self.options[key]['ymax']),**self.options[key]['attributes'])
          else:self.plt.axvspan(ast.literal_eval(self.options[key]['xmin']),ast.literal_eval(self.options[key]['xmax']), ymin=ast.literal_eval(self.options[key]['ymin']), ymax=ast.literal_eval(self.options[key]['ymax']))
      elif key == 'axes_box': 
        if   self.dim == 3: print('STREAM MANAGER: Warning -> axes_box not available in 3-D plots!!')
        elif self.dim == 2: self.plt.box(self.options[key][key])
      elif key == 'grid':
        if 'b' not in self.options[key].keys()  : self.options[key]['b'] = 'on'
        if self.options[key]['b'].lower() in ['on','t','true']: self.options[key]['b'] = 'off'
        elif self.options[key]['b'].lower() in ['off','f','false']: self.options[key]['b'] = 'off'
        if 'which' not in self.options[key].keys() : self.options[key]['which'] = 'major'
        if 'axis' not in self.options[key].keys() : self.options[key]['axis'] = 'both'
        if self.dim == 2:  
          if 'attributes' in self.options[key].keys(): self.plt.grid(ast.literal_eval(b =self.options[key]['b']),which = ast.literal_eval(self.options[key]['which']), axis=ast.literal_eval(self.options[key]['axis']),**self.options[key]['attributes'])
          else:self.plt.grid(b=self.options[key]['b'],which = (self.options[key]['which']), axis=(self.options[key]['axis']))
        elif self.dim == 3:
          if 'attributes' in self.options[key].keys(): self.plt3D.grid(b=ast.literal_eval(self.options[key]['b']),**self.options[key]['attributes'])
          else:self.plt3D.grid(b=ast.literal_eval(self.options[key]['b']))
      else:
        print('STREAM MANAGER: Warning -> Try to perform not-predifined action ' + key +'. If it does not work check manual and/or relavite matplotlib method specification.')
        command_args = ''
        for kk in self.options[key]:
          if kk != 'attributes' and kk != key:
            if command_args != '(': prefix = ','
            else: prefix = '' 
            try: command_args = prefix + command_args + kk + '=' + str(ast.literal_eval(self.options[key][kk]))
            except:command_args = prefix + command_args + kk + '="' + str(self.options[key][kk])+'"'  
        try:
          if self.dim == 2:  exec('self.plt.' + key + '(' + command_args + ')')
          elif self.dim == 3:exec('self.plt3D.' + key + '(' + command_args + ')')      
        except ValueError as ae: 
          raise Exception('STREAM MANAGER: ERROR <'+ae+'> -> in execution custom action "' + key + '" in Plot ' + self.name + '.\nSTREAM MANAGER: ERROR -> command has been called in the following way: ' + 'self.plt.' + key + '(' + command_args + ')')         

  ####################
  #  PUBLIC METHODS  #
  #################### 
  def localAddInitParams(self,tempDict):
    '''
      This method is called from the base function. It adds the initial characteristic intial params that need to be seen by the whole enviroment
      @ In, tempDict
      @ Out, tempDict
    '''
    tempDict['Plot is '] = str(self.dim)+'D'
    for index in range(len(self.sourceName)): tempDict['Source Name '+str(index)+' :'] = self.sourceName[index]

  def initialize(self,inDict):
    '''
    Function called to initialize the OutStream, linking it to the proper Data
    @ In, inDict -> Dictionary that contains all the instantiaced classes needed for the actual step
                    In this dictionary the data are looked for
    '''
    self.x_cordinates = []
    self.sourceName   = []
    for pltindex in range(len(self.options['plot_settings']['plot'])):
      if 'y' in self.options['plot_settings']['plot'][pltindex].keys(): self.y_cordinates = [] 
      if 'z' in self.options['plot_settings']['plot'][pltindex].keys(): self.z_cordinates = [] 
      if 'color_map' in self.options['plot_settings']['plot'][pltindex].keys(): self.color_map_coordinates = [] 
    for pltindex in range(len(self.options['plot_settings']['plot'])): 
      self.x_cordinates.append(self.options['plot_settings']['plot'][pltindex]['x'].split(',')) 
      self.sourceName.append(self.x_cordinates[pltindex][0].split('|')[0].strip())
      if 'y' in self.options['plot_settings']['plot'][pltindex].keys(): 
        self.y_cordinates.append(self.options['plot_settings']['plot'][pltindex]['y'].split(',')) 
        if self.y_cordinates[pltindex][0].split('|')[0] != self.sourceName[pltindex]: raise IOError('STREAM MANAGER: ERROR -> Every plot can be linked to one Data only. x_cord source is ' + self.sourceName[pltindex] + '. Got y_cord source is' + self.y_cordinates[pltindex][0].split('|')[0])
      if 'z' in self.options['plot_settings']['plot'][pltindex].keys(): 
        self.z_cordinates.append(self.options['plot_settings']['plot'][pltindex]['z'].split(',')) 
        if self.z_cordinates[pltindex][0].split('|')[0] != self.sourceName[pltindex]: raise IOError('STREAM MANAGER: ERROR -> Every plot can be linked to one Data only. x_cord source is ' + self.sourceName[pltindex] + '. Got z_cord source is' + self.z_cordinates[pltindex][0].split('|')[0])
      if 'color_map' in self.options['plot_settings']['plot'][pltindex].keys(): 
        self.color_map_coordinates.append(self.options['plot_settings']['plot'][pltindex]['color_map'].split(',')) 
        if self.color_map_coordinates[pltindex][0].split('|')[0] != self.sourceName[pltindex]: raise IOError('STREAM MANAGER: ERROR -> Every plot can be linked to one Data only. x_cord source is ' + self.sourceName[pltindex] + '. Got color_map_coordinates source is' + self.color_map_coordinates[pltindex][0].split('|')[0])
      for pltindex in range(len(self.options['plot_settings']['plot'])):
        if 'interpPointsY' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['interpPointsY'] = '20'
        if 'interpPointsX' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['interpPointsX'] = '20'
        if 'interpolation_type' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['interpolation_type'] = 'Rbflinear'
        elif self.options['plot_settings']['plot'][pltindex]['interpolation_type'] not in self.interpAvail: raise IOError('STREAM MANAGER: ERROR -> surface interpolation unknown. Available are :' + str(self.interpAvail))            
        if 'epsilon' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['epsilon'] = '2'
        if 'smooth' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['smooth'] = '0.0'
        if 'cmap' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['cmap'] = 'jet'
        elif self.options['plot_settings']['plot'][pltindex]['cmap'] not in self.mpl.cm.datad.keys(): raise('ERROR. The colorMap you specified does not exist... Available are ' + str(self.mpl.cm.datad.keys()))     
    self.numberAggregatedOS = len(self.options['plot_settings']['plot'])
    # initialize here the base class
    OutStreamManager.initialize(self,inDict)
    #execute actions (we execute the actions here also because we can perform a check at runtime!!
    self.__executeActions() 
  
  def localReadXML(self,xmlNode):
    '''
      This Function is called from the base class, It reads the parameters that belong to a plot block
      @ In, xmlNode
      @ Out, filled data structure (self)
    '''
    if not 'dim' in xmlNode.attrib.keys(): self.dim = 2
    else:                                  self.dim = int(xmlNode.attrib['dim'])
    foundPlot = False
    for subnode in xmlNode:
      # if actions, read actions block
      if subnode.tag in ['actions']: self.__readPlotActions(subnode)
      if subnode.tag in ['plot_settings']:
        self.options[subnode.tag] = {}
        self.options[subnode.tag]['plot'] = []
        for subsub in subnode:
          if subsub.tag == 'plot':
            tempDict = {}
            foundPlot = True
            for subsubsub in subsub:
              if subsubsub.tag != 'kwargs': tempDict[subsubsub.tag] = subsubsub.text
              else:
                tempDict['attributes'] = {}
                for sss in subsubsub: tempDict['attributes'][sss.tag] = sss.text       
            self.options[subnode.tag][subsub.tag].append(copy.deepcopy(tempDict))
          else: self.options[subnode.tag][subsub.tag] = subsub.text 
      if subnode.tag in 'title':
        self.options[subnode.tag] = {}
        for subsub in subnode: self.options[subnode.tag][subsub.tag] = subsub.text
        if 'text'     not in self.options[subnode.tag].keys(): self.options[subnode.tag]['text'    ] = xmlNode.attrib['name']
        if 'location' not in self.options[subnode.tag].keys(): self.options[subnode.tag]['location'] = 'center'   
      if subnode.tag == 'figure_properties':
        self.options[subnode.tag] = {}
        for subsub in subnode: self.options[subnode.tag][subsub.tag] = subsub.text         
    self.type = 'OutStreamPlot'
    if not 'plot_settings' in self.options.keys(): raise IOError('STREAM MANAGER: ERROR -> For plot named ' + self.name + ' the plot_settings block IS REQUIRED!!')
    if not foundPlot: raise IOError('STREAM MANAGER: ERROR -> For plot named'+ self.name + ', No plot section has been found in the plot_settings block!')
    self.outStreamTypes = []
    for pltindex in range(len(self.options['plot_settings']['plot'])):
      if not 'type' in self.options['plot_settings']['plot'][pltindex].keys(): raise IOError('STREAM MANAGER: ERROR -> For plot named'+ self.name + ', No plot type keyword has been found in the plot_settings/plot block!')
      else:
        if self.availableOutStreamTypes[self.dim].count(self.options['plot_settings']['plot'][pltindex]['type']) == 0: print('STREAM MANAGER: ERROR -> For plot named'+ self.name + ', type '+self.options['plot_settings']['plot'][pltindex]['type']+' is not among pre-defined plots! \n The OutstreamSystem will try to construct a call on the fly!!!') 
        self.outStreamTypes.append(self.options['plot_settings']['plot'][pltindex]['type']) 
    exec('self.mpl =  importlib.import_module("matplotlib")')
    print('STREAM MANAGER: matplotlib version is ' + str(self.mpl.__version__))
    if self.dim not in [2,3]: raise('STREAM MANAGER: ERROR -> This Plot interface is able to handle 2D-3D plot only')
    if not self.interactive or 'screen' not in self.options['how']['how']:
      self.interactive = False  # not needed interactive mode when no screen is requested
      self.mpl.use('Agg')       # set default backend to png
    exec('self.plt =  importlib.import_module("matplotlib.pyplot")')
    if self.interactive:self.plt.ion()
    if self.dim == 3:  exec('from mpl_toolkits.mplot3d import Axes3D as ' + 'Ax3D_' + self.name)
    if 'figure_properties' in self.options.keys():
      key = 'figure_properties'
      if 'figsize' not in self.options[key].keys():   self.options[key]['figsize'  ] = 'None' 
      if 'dpi' not in self.options[key].keys():       self.options[key]['dpi'      ] = 'None'
      if 'facecolor' not in self.options[key].keys(): self.options[key]['facecolor'] = 'None'
      if 'edgecolor' not in self.options[key].keys(): self.options[key]['edgecolor'] = 'None'
      if 'frameon' not in self.options[key].keys():   self.options[key]['frameon'  ] = 'True'
      elif self.options[key]['frameon'].lower() in ['t','true']: self.options[key]['frameon'] = 'True'
      elif self.options[key]['frameon'].lower() in ['f','false']: self.options[key]['frameon'] = 'False'           
      if 'attributes' in self.options[key].keys():  self.fig = self.plt.figure(self.name, figsize=ast.literal_eval(self.options[key]['figsize']), dpi=ast.literal_eval(self.options[key]['dpi']), facecolor=self.options[key]['facecolor'],edgecolor=self.options[key]['edgecolor'],frameon=ast.literal_eval(self.options[key]['frameon']),**self.options[key]['attributes'])
      else:  self.fig = self.plt.figure(self.name, figsize=ast.literal_eval(self.options[key]['figsize']), dpi=ast.literal_eval(self.options[key]['dpi']), facecolor=self.options[key]['facecolor'],edgecolor=self.options[key]['edgecolor'],frameon=ast.literal_eval(self.options[key]['frameon']))
    else: self.fig = self.plt.figure(self.name)
    if self.dim == 3: self.plt3D = self.fig.add_subplot(111, projection='3d')
  def addOutput(self):
    '''
    Function to show and/or save a plot 
    @ In,  None
    @ Out, None (Plot on the screen or on file/s) 
    ''' 
    # reactivate the figure
    self.plt.figure(self.name)
    # fill the x_values,y_values,z_values dictionaries
    if not self.__fillCoordinatesFromSource():
      print('STREAM MANAGER: Warning -> Nothing to Plot Yet... Returning!!!!')
      return
    self.counter += 1
    if self.counter > 1:
      if self.dim == 2: self.fig.clear()
      else: self.actPlot.remove()
    # execute the actions again (we just cleared the figure)
    self.__executeActions()
    # start plotting.... we are here fort that...aren't we?
    # loop over the plots that need to be included in this figure
    for pltindex in range(len(self.outStreamTypes)):
      # If the number of plots to be shown in this figure > 1, hold the old ones (They are going to be shown together... because unity is much better than separation)
      if len(self.outStreamTypes) > 1: self.plt.hold(True)
      if 'xlabel' not in self.options['plot_settings'].keys():
        if self.dim == 2  : self.plt.xlabel('x')
        elif self.dim == 3: self.plt3D.set_xlabel('x')
      else:
        if self.dim == 2  : self.plt.xlabel(self.options['plot_settings']['xlabel'])
        elif self.dim == 3: self.plt3D.set_xlabel(self.options['plot_settings']['xlabel'])
      if 'ylabel' not in self.options['plot_settings'].keys():
        if self.dim == 2  : self.plt.ylabel('y')
        elif self.dim == 3: self.plt3D.set_ylabel('y')
      else:
        if self.dim == 2  : self.plt.ylabel(self.options['plot_settings']['ylabel'])
        elif self.dim == 3: self.plt3D.set_ylabel(self.options['plot_settings']['ylabel'])          
      if 'zlabel' not in self.options['plot_settings'].keys():
        if self.dim == 2  : print('STREAM MANAGER: Warning -> zlabel keyword does not make sense in 2-D Plots!')
        elif self.dim == 3 and self.z_cordinates: self.plt3D.set_zlabel('z')
      elif self.dim == 3 and self.z_cordinates: self.plt3D.set_zlabel(self.options['plot_settings']['zlabel'])             
      # Let's start plotting
      #################
      #  SCATTER PLOT #
      ################# 
      if self.outStreamTypes[pltindex] == 'scatter':
        if 's' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['s'] = '20'
        if 'c' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['c'] = 'b'
        if 'marker' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['marker'] = 'o'   
        if 'alpha' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['alpha']='None'
        if 'linewidths' not in self.options['plot_settings']['plot'][pltindex].keys():  self.options['plot_settings']['plot'][pltindex]['linewidths'] = 'None'        
        for key in self.x_values[pltindex].keys():
          for x_index in range(len(self.x_values[pltindex][key])):
            for y_index in range(len(self.y_values[pltindex][key])):
              if self.dim == 2:
                if self.color_map_coordinates:
                  if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt.scatter(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],s=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['s']),c=self.color_map_values[pltindex][key],marker=(self.options['plot_settings']['plot'][pltindex]['marker']),alpha=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['alpha']),linewidths=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidths']),**self.options['plot_settings']['plot'][pltindex]['attributes'])
                  else: self.actPlot = self.plt.scatter(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],s=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['s']),c=self.color_map_values[pltindex][key],marker=(self.options['plot_settings']['plot'][pltindex]['marker']),alpha=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['alpha']),linewidths=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidths']))
                  m = self.mpl.cm.ScalarMappable(cmap=self.actPlot.cmap, norm=self.actPlot.norm)
                  m.set_array(self.color_map_values[pltindex][key])
                  actcm = self.plt.colorbar(m)                      
                  actcm.set_label(self.color_map_coordinates[pltindex][key-1].split('|')[-1].replace(')',''))                    
                else: 
                  if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt.scatter(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],s=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['s']),c=(self.options['plot_settings']['plot'][pltindex]['c']),marker=(self.options['plot_settings']['plot'][pltindex]['marker']),alpha=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['alpha']),linewidths=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidths']),**self.options['plot_settings']['plot'][pltindex]['attributes'])
                  else: self.actPlot = self.plt.scatter(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],s=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['s']),c=(self.options['plot_settings']['plot'][pltindex]['c']),marker=(self.options['plot_settings']['plot'][pltindex]['marker']),alpha=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['alpha']),linewidths=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidths']))
              elif self.dim == 3:
                for z_index in range(len(self.z_values[pltindex][key])):
                  if self.color_map_coordinates:
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.scatter(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],self.z_values[pltindex][key][z_index],rasterized= True,s=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['s']),c=self.color_map_values[pltindex][key],marker=(self.options['plot_settings']['plot'][pltindex]['marker']),alpha=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['alpha']),linewidths=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidths']),**self.options['plot_settings']['plot'][pltindex]['attributes'])
                    else: self.actPlot = self.plt3D.scatter(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],self.z_values[pltindex][key][z_index],s=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['s']),rasterized= True,c=self.color_map_values[pltindex][key],marker=(self.options['plot_settings']['plot'][pltindex]['marker']),alpha=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['alpha']),linewidths=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidths']))
                    m = self.mpl.cm.ScalarMappable(cmap=self.actPlot.cmap, norm=self.actPlot.norm)
                    m.set_array(self.color_map_values[pltindex][key])
                    actcm = self.plt.colorbar(m)                      
                    actcm.set_label(self.color_map_coordinates[pltindex][key-1].split('|')[-1].replace(')',''))      
                  else:
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.scatter(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],self.z_values[pltindex][key][z_index],rasterized= True,s=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['s']),c=(self.options['plot_settings']['plot'][pltindex]['c']),marker=(self.options['plot_settings']['plot'][pltindex]['marker']),alpha=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['alpha']),linewidths=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidths']),**self.options['plot_settings']['plot'][pltindex]['attributes'])
                    else: self.actPlot = self.plt3D.scatter(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],self.z_values[pltindex][key][z_index],s=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['s']),rasterized= True,c=(self.options['plot_settings']['plot'][pltindex]['c']),marker=(self.options['plot_settings']['plot'][pltindex]['marker']),alpha=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['alpha']),linewidths=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidths']))
      #################
      #   LINE PLOT   #
      #################       
      elif self.outStreamTypes[pltindex] == 'line':
        for key in self.x_values[pltindex].keys():
          for x_index in range(len(self.x_values[pltindex][key])):
            if self.color_map_coordinates: self.options['plot_settings']['plot'][pltindex]['interpPointsX'] = str(max(100,len(self.x_values[pltindex][key][x_index])))
            if self.x_values[pltindex][key][x_index].size < 2: xi = self.x_values[pltindex][key][x_index]
            else: xi = np.linspace(self.x_values[pltindex][key][x_index].min(),self.x_values[pltindex][key][x_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsX']))
            for y_index in range(len(self.y_values[pltindex][key])):
              if self.dim == 2:
                if ['nearest','linear','cubic'].count(self.options['plot_settings']['plot'][pltindex]['interpolation_type']) > 0:
                  if self.options['plot_settings']['plot'][pltindex]['interpolation_type'] != 'nearest' and self.y_values[pltindex][key][y_index].size > 2: yi = griddata((self.x_values[pltindex][key][x_index]), self.y_values[pltindex][key][y_index], (xi[None,:]), method=self.options['plot_settings']['plot'][pltindex]['interpolation_type'])
                  else: yi = griddata((self.x_values[pltindex][key][x_index]), self.y_values[pltindex][key][y_index], (xi[None,:]), method='nearest')
                else:
                  rbf = Rbf(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index],function=str(str(self.options['plot_settings']['plot'][pltindex]['interpolation_type']).replace('Rbf', '')),epsilon=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['epsilon']),smooth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['smooth']))
                  yi  = rbf(xi) 
                if self.color_map_coordinates:
                  # if a color map has been added, we use a scattered plot instead...
                  self.actPlot = self.plt.scatter(xi,yi,c=self.color_map_values[pltindex][key],marker='_')
                  m = self.mpl.cm.ScalarMappable(cmap=self.actPlot.cmap, norm=self.actPlot.norm)
                  m.set_array(self.color_map_values[pltindex][key])
                  actcm = self.plt.colorbar(m) 
                  actcm.set_label(self.color_map_coordinates[pltindex][key-1].split('|')[-1].replace(')',''))   
                else:
                  if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt.plot(xi,yi,**self.options['plot_settings']['plot'][pltindex]['attributes'])
                  else: self.actPlot = self.plt.plot(xi,yi)             
              elif self.dim == 3:
                if self.y_values[pltindex][key][y_index].size < 2: yi = self.y_values[pltindex][key][y_index]
                else: yi = np.linspace(self.y_values[pltindex][key][y_index].min(),self.y_values[pltindex][key][y_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsY']))
                for z_index in range(len(self.z_values[pltindex][key])):
                  if self.color_map_coordinates:
                    # if a color map has been added, we use a scattered plot instead...
                    self.actPlot = self.plt3D.scatter(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],self.z_values[pltindex][key][z_index],c=self.color_map_values[pltindex][key],marker='_')
                    m = self.mpl.cm.ScalarMappable(cmap=self.actPlot.cmap, norm=self.actPlot.norm)
                    m.set_array(self.color_map_values[pltindex][key])
                    actcm = self.plt.colorbar(m) 
                    actcm.set_label(self.color_map_coordinates[pltindex][key-1].split('|')[-1].replace(')',''))  
                  else:
                    if ['nearest','linear','cubic'].count(self.options['plot_settings']['plot'][pltindex]['interpolation_type']) > 0:
                      if self.options['plot_settings']['plot'][pltindex]['interpolation_type'] != 'nearest' and self.z_values[pltindex][key][z_index].size > 3: zi = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.z_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method=self.options['plot_settings']['plot'][pltindex]['interpolation_type'])
                      else: zi = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.z_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method='nearest')
                    else:
                      rbf = Rbf(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index], self.z_values[pltindex][key][z_index],function=str(str(self.options['plot_settings']['plot'][pltindex]['interpolation_type']).replace('Rbf', '')),epsilon=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['epsilon']),smooth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['smooth']))
                      zi  = rbf(xi, yi) 
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.plot(xi,yi,zi,**self.options['plot_settings']['plot'][pltindex]['attributes'])
                    else: self.actPlot = self.plt3D.plot(xi,yi,zi)
      ##################
      # HISTOGRAM PLOT #
      ##################                      
      elif self.outStreamTypes[pltindex] == 'histogram':
        if 'bins' in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['bins'] = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['bins'])
        else: self.options['plot_settings']['plot'][pltindex]['bins'] = '10'
        if 'normed' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['normed'] = 'False'
        else: self.options['plot_settings']['plot'][pltindex]['normed'] = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['normed'])
        if 'weights' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['weights'] = 'None'
        else: self.options['plot_settings']['plot'][pltindex]['weights'] = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['weights'])
        if 'cumulative' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['cumulative'] = 'False'
        else: self.options['plot_settings']['plot'][pltindex]['cumulative'] = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cumulative'])
        if 'histtype' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['histtype'] = 'bar'
        if 'align' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['align'] = 'mid'
        if 'orientation' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['orientation'] = 'vertical'                        
        if 'rwidth' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['rwidth'] = 'None'
        else: self.options['plot_settings']['plot'][pltindex]['rwidth'] = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rwidth'])
        if 'log' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['log'] = 'None'
        else: self.options['plot_settings']['plot'][pltindex]['log'] = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['log'])      
        if 'color' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['color'] = 'b'
        else: self.options['plot_settings']['plot'][pltindex]['color'] = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['color'])   
        if 'stacked' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['stacked'] = 'None'
        else: self.options['plot_settings']['plot'][pltindex]['stacked'] = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['stacked'])                 
        for key in self.x_values[pltindex].keys():
          for x_index in range(len(self.x_values[pltindex][key])):
            try: colorss = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['color'])
            except: colorss = self.options['plot_settings']['plot'][pltindex]['color']
            if self.dim == 2:  
              if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.plt.hist(self.x_values[pltindex][key][x_index], bins=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['bins']), normed=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['normed']), weights=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['weights']), 
                            cumulative=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cumulative']), histtype=self.options['plot_settings']['plot'][pltindex]['histtype'], align=self.options['plot_settings']['plot'][pltindex]['align'], 
                            orientation=self.options['plot_settings']['plot'][pltindex]['orientation'], rwidth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rwidth']), log=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['log']), 
                            color=colorss, stacked=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['stacked']), **self.options['plot_settings']['plot'][pltindex]['attributes'])
              else: self.plt.hist(self.x_values[pltindex][key][x_index], bins=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['bins']), normed=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['normed']), weights=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['weights']), 
                            cumulative=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cumulative']), histtype=self.options['plot_settings']['plot'][pltindex]['histtype'], align=self.options['plot_settings']['plot'][pltindex]['align'], 
                            orientation=self.options['plot_settings']['plot'][pltindex]['orientation'], rwidth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rwidth']), log=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['log']), 
                            color=colorss, stacked=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['stacked']))            
            elif self.dim == 3:
              for y_index in range(len(self.y_values[pltindex][key])):                
                hist, xedges, yedges = np.histogram2d(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index], bins=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['bins']))
                elements = (len(xedges) - 1) * (len(yedges) - 1)
                if 'x_offset' in self.options['plot_settings']['plot'][pltindex].keys(): xoffset = float(self.options['plot_settings']['plot'][pltindex]['x_offset'])
                else: xoffset = 0.0
                if 'y_offset' in self.options['plot_settings']['plot'][pltindex].keys(): yoffset = float(self.options['plot_settings']['plot'][pltindex]['y_offset'])
                else: yoffset = 0.0
                if 'dx' in self.options['plot_settings']['plot'][pltindex].keys(): dxs = float(self.options['plot_settings']['plot'][pltindex]['dx'])
                else: dxs = (self.x_values[pltindex][key][x_index].max() - self.x_values[pltindex][key][x_index].min())/self.options['plot_settings']['plot'][pltindex]['bins']
                if 'dy' in self.options['plot_settings']['plot'][pltindex].keys(): dys = float(self.options['plot_settings']['plot'][pltindex]['dy'])
                else: dys = (self.y_values[pltindex][key][y_index].max() - self.y_values[pltindex][key][y_index].min())/self.options['plot_settings']['plot'][pltindex]['bins']
                xpos, ypos = np.meshgrid(xedges[:-1]+xoffset, yedges[:-1]+yoffset)
                if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.bar3d(xpos.flatten(), ypos.flatten(), np.zeros(elements), dxs * np.ones_like(elements), dys * np.ones_like(elements), hist.flatten(), color=colorss, zsort='average', **self.options['plot_settings']['plot'][pltindex]['attributes'])
                else: self.actPlot = self.plt3D.bar3d(xpos.flatten(), ypos.flatten(), np.zeros(elements), dxs * np.ones_like(elements), dys * np.ones_like(elements), hist.flatten(), color=colorss, zsort='average')
      ##################
      #    STEM PLOT   #
      ##################                      
      elif self.outStreamTypes[pltindex] == 'stem':          
          if 'linefmt' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['linefmt'] = 'b-'
          if 'markerfmt' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['markerfmt'] = 'bo'
          if 'basefmt' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['basefmt'] = 'r-'
          for key in self.x_values[pltindex].keys():
            for x_index in range(len(self.x_values[pltindex][key])):
              for y_index in range(len(self.y_values[pltindex][key])):
                if self.dim == 2:
                  if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt.stem(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],linefmt=self.options['plot_settings']['plot'][pltindex]['linefmt'], markerfmt=self.options['plot_settings']['plot'][pltindex]['markerfmt'], basefmt=self.options['plot_settings']['plot'][pltindex]['linefmt'],**self.options['plot_settings']['plot'][pltindex]['attributes'])
                  else: self.actPlot = self.plt.stem(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],linefmt=self.options['plot_settings']['plot'][pltindex]['linefmt'], markerfmt=self.options['plot_settings']['plot'][pltindex]['markerfmt'], basefmt=self.options['plot_settings']['plot'][pltindex]['linefmt'])             
                elif self.dim == 3:
                  #it is a basic stem plot constructed using a standard line plot. For now we do not use the previous defined keywords...
                  for z_index in range(len(self.z_values[pltindex][key])):
                    for xx,yy,zz in zip(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],self.z_values[pltindex][key][z_index]): self.plt3D.plot([xx,xx],[yy,yy],[0,zz], '-')
      ##################
      #    STEP PLOT   #
      ##################                      
      elif self.outStreamTypes[pltindex] == 'step':      
        if self.dim == 2:
          if 'where' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['where'] = 'mid'
          for key in self.x_values[pltindex].keys():
            for x_index in range(len(self.x_values[pltindex][key])):
              if self.x_values[pltindex][key][x_index].size < 2: xi = self.x_values[pltindex][key][x_index]
              else: xi = np.linspace(self.x_values[pltindex][key][x_index].min(),self.x_values[pltindex][key][x_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsX']))
              for y_index in range(len(self.y_values[pltindex][key])):
                if ['nearest','linear','cubic'].count(self.options['plot_settings']['plot'][pltindex]['interpolation_type']) > 0:
                  if self.options['plot_settings']['plot'][pltindex]['interpolation_type'] != 'nearest' and self.y_values[pltindex][key][y_index].size > 2: yi = griddata((self.x_values[pltindex][key][x_index]), self.y_values[pltindex][key][y_index], (xi[None,:]), method=self.options['plot_settings']['plot'][pltindex]['interpolation_type'])
                  else: yi = griddata((self.x_values[pltindex][key][x_index]), self.y_values[pltindex][key][y_index], (xi[None,:]), method='nearest')
                else:
                  rbf = Rbf(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index],function=str(str(self.options['plot_settings']['plot'][pltindex]['interpolation_type']).replace('Rbf', '')),epsilon=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['epsilon']),smooth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['smooth']))
                  yi  = rbf(xi) 
                if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt.step(xi,yi,where=self.options['plot_settings']['plot'][pltindex]['where'],**self.options['plot_settings']['plot'][pltindex]['attributes'])
                else: self.actPlot = self.plt.step(xi,yi,where=self.options['plot_settings']['plot'][pltindex]['where'])
        elif self.dim == 3: 
          print('STREAM MANAGER: step Plot not available in 3D')
          return
      ########################
      #    PSEUDOCOLOR PLOT  #
      ########################           
      elif self.outStreamTypes[pltindex] == 'pseudocolor':
        if self.dim == 2:
          for key in self.x_values[pltindex].keys():
            for x_index in range(len(self.x_values[pltindex][key])):
              xi = np.linspace(self.x_values[pltindex][key][x_index].min(),self.x_values[pltindex][key][x_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsX']))
              for y_index in range(len(self.y_values[pltindex][key])):
                yi = np.linspace(self.y_values[pltindex][key][y_index].min(),self.y_values[pltindex][key][y_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsY']))
                xig, yig = np.meshgrid(xi, yi)           
                if not self.color_map_coordinates: 
                  print('STREAM MANAGER: pseudocolor Plot needs coordinates for color map... Returning without plotting')
                  return
                for z_index in range(len(self.color_map_values[pltindex][key])):
                  if ['nearest','linear','cubic'].count(self.options['plot_settings']['plot'][pltindex]['interpolation_type']) > 0:
                    if self.options['plot_settings']['plot'][pltindex]['interpolation_type'] != 'nearest' and self.color_map_values[pltindex][key][z_index].size > 3: Ci = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.color_map_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method=self.options['plot_settings']['plot'][pltindex]['interpolation_type'])
                    else: Ci = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.color_map_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method='nearest')
                  else:
                    rbf = Rbf(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index], self.color_map_values[pltindex][key][z_index],function=str(str(self.options['plot_settings']['plot'][pltindex]['interpolation_type']).replace('Rbf', '')),epsilon=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['epsilon']),smooth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['smooth']))
                    Ci  = rbf(xig, yig)   
                  if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt.pcolormesh(xig,yig,Ci,**self.options['plot_settings']['plot'][pltindex]['attributes'])
                  else: self.actPlot  = self.plt.pcolormesh(xig,yig,Ci)
                  m = self.mpl.cm.ScalarMappable(cmap=self.actPlot.cmap, norm=self.actPlot.norm)
                  m.set_array(Ci)
                  actcm = self.plt.colorbar(m) 
                  actcm.set_label(self.color_map_coordinates[pltindex][key-1].split('|')[-1].replace(')',''))                    
        elif self.dim == 3: 
          print('STREAM MANAGER: pseudocolor Plot is considered a 2D plot, not a 3D!')
          return
      ########################
      #     SURFACE PLOT     #
      ########################           
      elif self.outStreamTypes[pltindex] == 'surface':
        if self.dim == 2: 
          print('STREAM MANAGER: surface Plot is NOT available for 2D plots, IT IS A 2D!')
          return
        elif self.dim == 3:
          if 'rstride' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['rstride'] = '1'
          if 'cstride' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['cstride'] = '1'
          if 'antialiased' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['antialiased']='False'
          if 'linewidth' not in self.options['plot_settings']['plot'][pltindex].keys():  self.options['plot_settings']['plot'][pltindex]['linewidth'] = '0'
          for key in self.x_values[pltindex].keys():
            for x_index in range(len(self.x_values[pltindex][key])):
              xi = np.linspace(self.x_values[pltindex][key][x_index].min(),self.x_values[pltindex][key][x_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsX']))
              for y_index in range(len(self.y_values[pltindex][key])):
                yi = np.linspace(self.y_values[pltindex][key][y_index].min(),self.y_values[pltindex][key][y_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsY']))
                xig, yig = np.meshgrid(xi, yi)
                for z_index in range(len(self.z_values[pltindex][key])):        
                  if ['nearest','linear','cubic'].count(self.options['plot_settings']['plot'][pltindex]['interpolation_type']) > 0:
                    if self.color_map_coordinates:
                      if self.options['plot_settings']['plot'][pltindex]['interpolation_type'] != 'nearest' and self.color_map_values[pltindex][key][z_index].size > 3: Ci = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.color_map_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method=self.options['plot_settings']['plot'][pltindex]['interpolation_type'])
                      else: Ci = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.color_map_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method='nearest')           
                    if self.options['plot_settings']['plot'][pltindex]['interpolation_type'] != 'nearest' and self.z_values[pltindex][key][z_index].size > 3: zi = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.z_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method=self.options['plot_settings']['plot'][pltindex]['interpolation_type'])
                    else: zi = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.z_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method='nearest')
                  else:
                    if self.color_map_coordinates:
                      rbfc = Rbf(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index], self.color_map_values[pltindex][key][z_index],function=str(str(self.options['plot_settings']['plot'][pltindex]['interpolation_type']).replace('Rbf', '')),epsilon=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['epsilon']),smooth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['smooth']))
                      Ci  = rbfc(xig, yig)
                      print(Ci)    
                    rbf = Rbf(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index], self.z_values[pltindex][key][z_index],function=str(str(self.options['plot_settings']['plot'][pltindex]['interpolation_type']).replace('Rbf', '')),epsilon=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['epsilon']),smooth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['smooth']))
                    zi  = rbf(xig, yig) 
                  if self.color_map_coordinates:
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.plot_surface(xig,yig,zi, rstride = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rstride']), cstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cstride']),facecolors=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap'])(Ci),cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']),linewidth= ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidth']),antialiased=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['antialiased']),**self.options['plot_settings']['plot'][pltindex]['attributes'])    
                    else: self.actPlot = self.plt3D.plot_surface(xig,yig,zi,rstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rstride']), cstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cstride']),facecolors=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap'])(Ci),cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']),linewidth= ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidth']),antialiased=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['antialiased']))                    
                    m = self.mpl.cm.ScalarMappable(cmap=self.actPlot.cmap, norm=self.actPlot.norm)
                    m.set_array(Ci)
                    actcm = self.plt.colorbar(m) 
                    actcm.set_label(self.color_map_coordinates[pltindex][key-1].split('|')[-1].replace(')',''))                                 
                  else:
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.plot_surface(xig,yig,zi, rstride = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rstride']), cstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cstride']),cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']),linewidth= ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidth']),antialiased=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['antialiased']),**self.options['plot_settings']['plot'][pltindex]['attributes'])    
                    else: self.actPlot = self.plt3D.plot_surface(xig,yig,zi,rstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rstride']), cstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cstride']),cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']),linewidth= ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['linewidth']),antialiased=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['antialiased'])) 
      ########################
      #   TRI-SURFACE PLOT   #
      ########################           
      elif self.outStreamTypes[pltindex] == 'tri-surface':
        if self.dim == 2: 
          print('STREAM MANAGER: TRI-surface Plot is NOT available for 2D plots, IT IS A 2D!')
          return
        elif self.dim == 3:
          if 'color' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['color'] = 'b'
          if 'shade' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['shade']='False'
          for key in self.x_values[pltindex].keys():
            for x_index in range(len(self.x_values[pltindex][key])):
              for y_index in range(len(self.y_values[pltindex][key])):
                for z_index in range(len(self.z_values[pltindex][key])):
                  if self.color_map_coordinates:
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.plot_trisurf(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],self.z_values[pltindex][key][z_index], color = self.options['plot_settings']['plot'][pltindex]['color'],cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']),shade= ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['shade']),**self.options['plot_settings']['plot'][pltindex]['attributes'])    
                    else: self.actPlot = self.plt3D.plot_trisurf(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],self.z_values[pltindex][key][z_index], color = self.options['plot_settings']['plot'][pltindex]['color'],cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']),shade= ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['shade']))
                  else:
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.plot_trisurf(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],self.z_values[pltindex][key][z_index], color = self.options['plot_settings']['plot'][pltindex]['color'],cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']),shade= ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['shade']),**self.options['plot_settings']['plot'][pltindex]['attributes'])    
                    else: self.actPlot = self.plt3D.plot_trisurf(self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index],self.z_values[pltindex][key][z_index], color = self.options['plot_settings']['plot'][pltindex]['color'],cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']),shade= ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['shade']))
      ########################
      #    WIREFRAME  PLOT   #
      ########################           
      elif self.outStreamTypes[pltindex] == 'wireframe':
        if self.dim == 2: 
          print('STREAM MANAGER: wireframe Plot is NOT available for 2D plots, IT IS A 2D!')
          return
        elif self.dim == 3:
          if 'rstride' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['rstride'] = '1'
          if 'cstride' not in self.options['plot_settings']['plot'][pltindex].keys(): self.options['plot_settings']['plot'][pltindex]['cstride'] = '1'
        
          for key in self.x_values[pltindex].keys():
            for x_index in range(len(self.x_values[pltindex][key])):
              xi = np.linspace(self.x_values[pltindex][key][x_index].min(),self.x_values[pltindex][key][x_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsX']))
              for y_index in range(len(self.y_values[pltindex][key])):
                yi = np.linspace(self.y_values[pltindex][key][y_index].min(),self.y_values[pltindex][key][y_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsY']))
                xig, yig = np.meshgrid(xi, yi)
                for z_index in range(len(self.z_values[pltindex][key])):        
                  if ['nearest','linear','cubic'].count(self.options['plot_settings']['plot'][pltindex]['interpolation_type']) > 0:
                    if self.color_map_coordinates:
                      if self.options['plot_settings']['plot'][pltindex]['interpolation_type'] != 'nearest' and self.color_map_values[pltindex][key][z_index].size > 3: Ci = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.color_map_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method=self.options['plot_settings']['plot'][pltindex]['interpolation_type'])
                      else: Ci = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.color_map_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method='nearest')           
                    if self.options['plot_settings']['plot'][pltindex]['interpolation_type'] != 'nearest' and self.z_values[pltindex][key][z_index].size > 3: zi = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.z_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method=self.options['plot_settings']['plot'][pltindex]['interpolation_type'])
                    else: zi = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.z_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method='nearest')
                  else:
                    if self.color_map_coordinates:
                      rbfc = Rbf(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index], self.color_map_values[pltindex][key][z_index],function=str(str(self.options['plot_settings']['plot'][pltindex]['interpolation_type']).replace('Rbf', '')),epsilon=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['epsilon']),smooth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['smooth']))
                      Ci  = rbfc(xig, yig)    
                    rbf = Rbf(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index], self.z_values[pltindex][key][z_index],function=str(str(self.options['plot_settings']['plot'][pltindex]['interpolation_type']).replace('Rbf', '')),epsilon=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['epsilon']),smooth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['smooth']))
                    zi  = rbf(xig, yig) 
                  if self.color_map_coordinates:
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.plot_wireframe(xig,yig,zi, rstride = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rstride']), color=Ci, cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']), cstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cstride']),**self.options['plot_settings']['plot'][pltindex]['attributes'])    
                    else: self.actPlot = self.plt3D.plot_wireframe(xig,yig,zi,rstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rstride']), color=Ci, cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']),cstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cstride'])) 
                    m = self.mpl.cm.ScalarMappable(cmap=self.actPlot.cmap, norm=self.actPlot.norm)
                    m.set_array(Ci)
                    actcm = self.plt.colorbar(m) 
                    actcm.set_label(self.color_map_coordinates[pltindex][key-1].split('|')[-1].replace(')',''))                     
                  else:
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.plot_wireframe(xig,yig,zi, rstride = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rstride']), cstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cstride']),**self.options['plot_settings']['plot'][pltindex]['attributes'])    
                    else: self.actPlot = self.plt3D.plot_wireframe(xig,yig,zi,rstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rstride']), cstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cstride'])) 

                for z_index in range(len(self.z_values[pltindex][key])):
                  if ['nearest','linear','cubic'].count(self.options['plot_settings']['plot'][pltindex]['interpolation_type']) > 0:
                    if self.options['plot_settings']['plot'][pltindex]['interpolation_type'] != 'nearest' and self.z_values[pltindex][key][z_index].size > 3: zi = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.z_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method=self.options['plot_settings']['plot'][pltindex]['interpolation_type'])
                    else: zi = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.z_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method='nearest')
                  else:
                    rbf = Rbf(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index], self.z_values[pltindex][key][z_index],function=str(str(self.options['plot_settings']['plot'][pltindex]['interpolation_type']).replace('Rbf', '')),epsilon=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['epsilon']),smooth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['smooth']))
                    zi  = rbf(xig, yig) 
                  if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.plot_wireframe(xig,yig,zi, rstride = ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rstride']), cstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cstride']),**self.options['plot_settings']['plot'][pltindex]['attributes'])    
                  else: self.actPlot = self.plt3D.plot_wireframe(xig,yig,zi,rstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['rstride']), cstride=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['cstride'])) 
      ########################
      #     CONTOUR   PLOT   #
      ########################           
      elif self.outStreamTypes[pltindex] == 'contour' or self.outStreamTypes[pltindex] == 'filled_contour':
        if self.dim == 2:
          if 'number_bins' in self.options['plot_settings']['plot'][pltindex].keys(): nbins = int(self.options['plot_settings']['plot'][pltindex]['number_bins'])
          else: nbins = 5
          for key in self.x_values[pltindex].keys():
            if not self.color_map_coordinates: 
              print('STREAM MANAGER: '+self.outStreamTypes[pltindex]+' Plot needs coordinates for color map... Returning without plotting')
              return
            for x_index in range(len(self.x_values[pltindex][key])):
              xi = np.linspace(self.x_values[pltindex][key][x_index].min(),self.x_values[pltindex][key][x_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsX']))
              for y_index in range(len(self.y_values[pltindex][key])):
                yi = np.linspace(self.y_values[pltindex][key][y_index].min(),self.y_values[pltindex][key][y_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsY']))
                xig, yig = np.meshgrid(xi, yi)
                for z_index in range(len(self.color_map_values[pltindex][key])):        
                  if ['nearest','linear','cubic'].count(self.options['plot_settings']['plot'][pltindex]['interpolation_type']) > 0:
                    if self.options['plot_settings']['plot'][pltindex]['interpolation_type'] != 'nearest' and self.color_map_values[pltindex][key][z_index].size > 3: Ci = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.color_map_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method=self.options['plot_settings']['plot'][pltindex]['interpolation_type'])
                    else: Ci = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.color_map_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method='nearest')           
                  else:    
                    rbf = Rbf(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index], self.color_map_values[pltindex][key][z_index],function=str(str(self.options['plot_settings']['plot'][pltindex]['interpolation_type']).replace('Rbf', '')),epsilon=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['epsilon']),smooth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['smooth']))
                    Ci  = rbf(xig, yig) 
                  if self.outStreamTypes[pltindex] == 'contour':
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt.contour(xig,yig,Ci,nbins,**self.options['plot_settings']['plot'][pltindex]['attributes'])
                    else: self.actPlot  = self.plt.contour(xig,yig,Ci,nbins)
                  else:
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt.contourf(xig,yig,Ci,nbins,**self.options['plot_settings']['plot'][pltindex]['attributes'])
                    else: self.actPlot  = self.plt.contourf(xig,yig,Ci,nbins)
                  self.plt.clabel(self.actPlot, inline=1, fontsize=10)
                  self.plt.colorbar(self.actPlot, shrink=0.8, extend='both')
        elif self.dim == 3: 
          print('STREAM MANAGER: contour/filled_contour is a 2-D plot, where x,y are the surface coordinates and color_map vector is the array to visualize!\n               contour3D/filled_contour3D are 3-D! ')
          return
      elif self.outStreamTypes[pltindex] == 'contour3D' or self.outStreamTypes[pltindex] == 'filled_contour3D':
        if self.dim == 2: 
          print('STREAM MANAGER: contour3D/filled_contour3D Plot is NOT available for 2D plots, IT IS A 2D! Check "contour/filled_contour"!')
          return 
        elif self.dim == 3:
          if 'number_bins' in self.options['plot_settings']['plot'][pltindex].keys(): nbins = int(self.options['plot_settings']['plot'][pltindex]['number_bins'])
          else: nbins = 5
          if 'extend3D' in self.options['plot_settings']['plot'][pltindex].keys(): ext3D = bool(self.options['plot_settings']['plot'][pltindex]['extend3D'])
          else: ext3D = False
          for key in self.x_values[pltindex].keys():
            for x_index in range(len(self.x_values[pltindex][key])):
              xi = np.linspace(self.x_values[pltindex][key][x_index].min(),self.x_values[pltindex][key][x_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsX']))
              for y_index in range(len(self.y_values[pltindex][key])):
                yi = np.linspace(self.y_values[pltindex][key][y_index].min(),self.y_values[pltindex][key][y_index].max(),ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['interpPointsY']))
                xig, yig = np.meshgrid(xi, yi)
                for z_index in range(len(self.color_map_values[pltindex][key])):        
                  if ['nearest','linear','cubic'].count(self.options['plot_settings']['plot'][pltindex]['interpolation_type']) > 0:
                    if self.options['plot_settings']['plot'][pltindex]['interpolation_type'] != 'nearest' and self.color_map_values[pltindex][key][z_index].size > 3: Ci = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.color_map_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method=self.options['plot_settings']['plot'][pltindex]['interpolation_type'])
                    else: Ci = griddata((self.x_values[pltindex][key][x_index],self.y_values[pltindex][key][y_index]), self.z_values[pltindex][key][z_index], (xi[None,:], yi[:,None]), method='nearest')           
                  else:    
                    rbf = Rbf(self.x_values[pltindex][key][x_index], self.y_values[pltindex][key][y_index], self.z_values[pltindex][key][z_index],function=str(str(self.options['plot_settings']['plot'][pltindex]['interpolation_type']).replace('Rbf', '')),epsilon=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['epsilon']),smooth=ast.literal_eval(self.options['plot_settings']['plot'][pltindex]['smooth']))
                    Ci  = rbf(xig, yig) 
                  if self.outStreamTypes[pltindex] == 'contour3D':
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.contour3D(xig,yig,Ci,nbins,extend3d=ext3D, cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']),**self.options['plot_settings']['plot'][pltindex]['attributes'])
                    else: self.actPlot  = self.plt3D.contour3D(xig,yig,Ci,nbins,extend3d=ext3D, cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']))
                  else:
                    if 'attributes' in self.options['plot_settings']['plot'][pltindex].keys(): self.actPlot = self.plt3D.contourf3D(xig,yig,Ci,nbins,extend3d=ext3D, cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']),**self.options['plot_settings']['plot'][pltindex]['attributes'])
                    else: self.actPlot  = self.plt3D.contourf3D(xig,yig,Ci,nbins,extend3d=ext3D, cmap=self.mpl.cm.get_cmap(name=self.options['plot_settings']['plot'][pltindex]['cmap']))
                  self.plt.clabel(self.actPlot, inline=1, fontsize=10)
                  self.plt.colorbar(self.actPlot, shrink=0.8, extend='both')                    
      else:
        # Let's try to "write" the code for the plot on the fly
        print('STREAM MANAGER: Warning -> Try to create a not-predifined plot of type ' + self.outStreamTypes[pltindex] +'. If it does not work check manual and/or relavite matplotlib method specification.')
        command_args = ''
        for kk in self.options['plot_settings']['plot'][pltindex]:
          if kk != 'attributes' and kk != self.outStreamTypes[pltindex]:
            if command_args != '(': prefix = ','
            else: prefix = '' 
            try: command_args = prefix + command_args + kk + '=' + str(ast.literal_eval(self.options['plot_settings']['plot'][pltindex][kk]))
            except:command_args = prefix + command_args + kk + '="' + str(self.options['plot_settings']['plot'][pltindex][kk])+'"'  
        try:
          if self.dim == 2:  exec('self.actPlot = self.plt.' + self.outStreamTypes[pltindex] + '(' + command_args + ')')
          elif self.dim == 3:exec('self.actPlot = self.plt3D.' + self.outStreamTypes[pltindex] + '(' + command_args + ')')      
        except ValueError as ae: 
          raise Exception('STREAM MANAGER: ERROR <'+ae+'> -> in execution custom plot "' + self.outStreamTypes[pltindex] + '" in Plot ' + self.name + '.\nSTREAM MANAGER: ERROR -> command has been called in the following way: ' + 'self.plt.' + self.outStreamTypes[pltindex] + '(' + command_args + ')')         
    # SHOW THE PICTURE
    if 'screen' in self.options['how']['how'].split(','): 
      if self.dim == 2 or 'pseudocolor' in self.outStreamTypes: self.fig.canvas.draw()
      else:self.plt.draw()
      if not self.interactive:self.plt.show()
    for i in range(len(self.options['how']['how'].split(','))):
      if self.options['how']['how'].split(',')[i].lower() != 'screen':
        if not self.overwrite: prefix = str(self.counter) + '-'
        else: prefix = ''
        self.plt.savefig(prefix + self.name+'_' + str(self.outStreamTypes).replace("'", "").replace("[", "").replace("]", "").replace(",", "-").replace(" ", "") +'.'+self.options['how']['how'].split(',')[i], format=self.options['how']['how'].split(',')[i])        

class OutStreamPrint(OutStreamManager):
  def __init__(self):
    self.type = 'OutStreamPrint'
    self.availableOutStreamTypes = ['csv']
    OutStreamManager.__init__(self)
    self.sourceName   = []
    self.sourceData   = None
    self.variables    = None 

  def localAddInitParams(self,tempDict):
    for index in range(len(self.sourceName)): tempDict['Source Name '+str(index)+' :'] = self.sourceName[index]
    if self.variables:
      for index in range(len(self.variables)): tempDict['Variable Name '+str(index)+' :'] = self.variables[index]
     
  def initialize(self,inDict):
    # the linking to the source is performed in the base class initialize method
    OutStreamManager.initialize(self,inDict)

  def localReadXML(self,xmlNode):
    self.type = 'OutStreamPrint'
    for subnode in xmlNode: 
      if subnode.tag == 'source': self.sourceName = subnode.text.split(',')
      else:self.options[subnode.tag] = subnode.text
    if 'type' not in self.options.keys(): raise('STREAM MANAGER: ERROR -> type tag not present in Print block called '+ self.name)
    if self.options['type'] not in self.availableOutStreamTypes : raise('STREAM MANAGER: ERROR -> Print type ' + self.options['type'] + ' not available yet. ')
    if 'variables' in self.options.keys(): self.variables = self.options['variables']
  def addOutput(self):
    if self.variables: dictOptions = {'filenameroot':self.name,'variables':self.variables}
    else             : dictOptions = {'filenameroot':self.name}
    for index in range(len(self.sourceName)): 
      if not self.sourceData[index].isItEmpty(): self.sourceData[index].printCSV(dictOptions)
     
'''
 Interface Dictionary (factory) (private)
'''
__base                     = 'OutStreamManager'
__interFaceDict            = {}
__interFaceDict['Plot'   ] = OutStreamPlot
__interFaceDict['Print'  ] = OutStreamPrint
__knownTypes              = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  '''
  function used to generate a OutStream class
  @ In, Type : OutStream type
  @ Out,Instance of the Specialized OutStream class
  '''
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)  
  



