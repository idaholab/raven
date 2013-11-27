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

  def initialize(self,inDict):
    '''
    Function to link the source object to the outstream object (i.e. the Data)
    '''
    raise NotYetImplemented('Li Mortacci!!!!!! Non implementataaaa!!!!')
    
    
  def readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the got inputs 
    @ In, xmlNode    : Xml element node
    @ Out, None
    '''
    BaseType.readMoreXML(self,xmlNode)
    if 'interactive' in xmlNode.attrib.keys(): self.interactive = bool(xmlNode.attrib['interactive'])
    if xmlNode.attrib['type'] not in self.availableOutStreamTypes: raise IOError('STREAM MANAGER: ERROR -> type "'+ node.attrib['type']+'" not available!')
    else: self.outStreamType = xmlNode.attrib['type']
  
#     for node in xmlNode:
#       self.num_active_outstreams += 1
#       if node.tag == 'Assemble': pass
#       elif node.tag == 'Print':
#         if node.attrib['name'] in self.outstreamDict[node.tag].keys(): raise IOError('STREAM MANAGER: ERROR -> print named "' + node.attrib['name'] +'" appears multiple times!')
#         else:self.outstreamDict[node.tag][node.attrib['name']]={node.tag:[node.attrib['type']]}
#         if node.attrib['type'] not in self.__availableOutStreamTypes[node.tag]: raise IOError('STREAM MANAGER: ERROR -> print type "'+ node.attrib['type']+'" not available!')
#         self.outstreamDict[node.tag][node.attrib['name']]= OutStreamPrint(node.attrib['name'],node.attrib['type'])
#         self.outstreamDict[node.tag][node.attrib['name']].readMoreXML(node)    
#       elif node.tag == 'Plot':
#         if node.attrib['name'] in self.outstreamDict[node.tag].keys(): raise IOError('STREAM MANAGER: ERROR -> plot named "' + node.attrib['name'] +'" appears multiple times')
#         if node.attrib['type'] not in self.__availableOutStreamTypes[node.tag]: raise IOError('STREAM MANAGER: ERROR -> plot type "'+ node.attrib['type']+'" not available!')
#         self.outstreamDict[node.tag][node.attrib['name']]= OutStreamPlot(node.attrib['name'],node.attrib['type'],node.attrib['dim'])
#         self.outstreamDict[node.tag][node.attrib['name']].readMoreXML(node)  
#       else: raise IOError('STREAM MANAGER: ERROR -> tag "' + node.tag +'" unknown')
    

  def addInitParams(self,tempDict):
    '''
    Function adds the initial parameter in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    '''
    return tempDict

  def finalize(self):
    '''
    Function to finalize the outstream.Each outstream specialized class must implement it
    @ In, None
    @ Out, None 
    '''
    pass 

  def addOutput(self,toLoadFrom):
    '''
    Function to add a new output source (for example a CSV file or a HDF5 object)
    @ In, toLoadFrom, source object
    @ Out, None 
    '''
    # this function adds the file name/names to the
    # filename list
    
#  def getInpParametersValues(self):
#    return self.inpParametersValues  
#
#  def getOutParametersValues(self):
#    return self.outParametersValues 

  def retrieveHistories(self):
    '''
    Function to retrieve histories from th toLoadFromList object
    @ In, None
    @ Out, None 
    '''
    # Check type of source
    try:
      if self.toLoadFromList[0].type == "HDF5":
        # HDF5 database
        # Retrieve ending histories' names from the database
        endGroupNames = self.toLoadFromList[0].getEndingGroupNames()
        # Retrieve the histories
        for index in xrange(len(endGroupNames)):
          if not endGroupNames[index] in self.alreadyRead:
            self.histories[endGroupNames[index]] = self.toLoadFromList[0].returnHistory({'history':endGroupNames[index],'filter':'whole'})
            self.alreadyRead.append(endGroupNames[index])
    except AttributeError:
      # loading from file (csv) 
      # Retrieve histories from CSV files
      for index in xrange(len(self.toLoadFromList)):
        groupname = self.toLoadFromList[index].split('~')[1]
        if not groupname in self.alreadyRead:
          # open file
          myFile = open (self.toLoadFromList[index],'rb')
          # read the field names
          all_field_names = myFile.readline().split(b',')
          # load the table data (from the csv file) into a numpy nd array 
          data = np.loadtxt(myFile,dtype='float',delimiter=',',ndmin=2)
          # close file
          myFile.close()  
          self.histories[groupname] = (data,{'headers':all_field_names})
          self.alreadyRead.append(groupname)
    return

  def getParam(self,typeVar,keyword):
    '''
    Function to get a Parameter in this function
    @ In, typeVar : Variable type (string)
    @ In, keyword: Keyword to retrieve 
    @ Out,param  : Requested parameter
    '''
    pass

class OutStreamPlot(OutStreamManager):
  def __init__(self):
    self.availableOutStreamTypes = ['scatter','line','surface','histogram','stem','step','polar','pseudocolor']
    OutStreamManager.__init__(self)
    self.sourceName   = None
    self.sourceData   = None
    self.x_cordinates = None
    self.y_cordinates = None
    self.z_cordinates = None
    self.plotSettings = {}


  def initialize(self,inDict):
    '''
    Function called to initialize the OutStream linking it to the proper Data
    '''
    self.x_cordinates = self.options['plot_settings']['x'].split(',')
    self.sourceName = self.x_cordinates[0].split('|')[0].strip()
    if 'y' in self.options['plot_settings'].keys(): 
      self.y_cordinates = self.options['plot_settings']['y'].split(',')
      if self.y_cordinates[0].split('|')[0] != self.sourceName: raise IOError('STREAM MANAGER: ERROR -> Every plot can be linked to one Data only. x_cord source is ' + self.sourceName + '. Got y_cord source is' + self.y_cordinates[0].split('|')[0])
    if 'z' in self.options['plot_settings'].keys(): 
      self.z_cordinates = self.options['plot_settings']['z'].split(',')
      if self.z_cordinates[0].split('|')[0] != self.sourceName: raise IOError('STREAM MANAGER: ERROR -> Every plot can be linked to one Data only. x_cord source is ' + self.sourceName + '. Got z_cord source is' + self.z_cordinates[0].split('|')[0])
    foundData = False
    for output in inDict['Output']:
      if output.name.strip() == self.sourceName:
        self.sourceData = output
        foundData = True
    if not foundData:
      for inp in inDict['Input']:
        if type(inp) != str:
          if inp.name.strip() == self.sourceName:
            self.sourceData = inp
            foundData = True  
    if not foundData: raise IOError('STREAM MANAGER: ERROR -> the Data named ' + self.sourceName + 'has not been found!!!!')
    # retrieve all the other plot settings (plot dependent) 
    for key in self.options['plot_settings'].keys():
      if key not in ['x','y','z']: self.plotSettings[key] = self.options['plot_settings'][key]
    #execute actions
    self.plt.ioff()
    self.__executeActions()    
  def __readPlotActions(self,snode):
    #if snode.find('how') is not None: self.options[snode.tag]['how'] = snode.find('how').text.lower()
    #else: self.options[snode.tag]['how'] = 'screen'
    for node in snode:
      self.options[node.tag] = {}
      if any(node.attrib):
        self.options[node.tag]['attributes'] = {}
        for key in node.attrib.keys(): 
          try: self.options[node.tag]['attributes'][key] = ast.literal_eval(node.attrib[key])
          except AttributeError: self.options[node.tag]['attributes'][key] = node.attrib[key]
      if len(node):
        for subnode in node: self.options[node.tag][subnode.tag] = subnode.text
      elif node.text: 
        if node.text.strip(): self.options[node.tag][node.tag] = node.text
    if 'how' not in self.options.keys(): self.options['how']={'how':'screen'} 

  def __fillCoordinatesFromSource(self):
    if self.sourceData.type.strip() not in 'Histories': 
      self.x_values = {1:[]}
      if self.y_cordinates: self.y_values = {1:[]}
      if self.z_cordinates: self.z_values = {1:[]}
      for i in range(len(self.x_cordinates)): 
        self.x_values[1].append(self.sourceData.getParam(self.x_cordinates[i].split('|')[1],self.x_cordinates[i].split('|')[2]))
      if self.y_cordinates:
        for i in range(len(self.y_cordinates)): self.y_values[1].append(self.sourceData.getParam(self.y_cordinates[i].split('|')[1],self.y_cordinates[i].split('|')[2]))
      if self.z_cordinates:
        for i in range(len(self.z_cordinates)): self.z_values[1].append(self.sourceData.getParam(self.z_cordinates[i].split('|')[1],self.z_cordinates[i].split('|')[2]))
    else:
      self.x_values = {}
      if self.y_cordinates: self.y_values = {}
      if self.z_cordinates: self.z_values = {}
      for key in self.sourceData.getInpParametersValues().keys(): 
        self.x_values[key] = []
        if self.y_cordinates: self.y_values[key] = []
        if self.z_cordinates: self.z_values[key] = []
        for i in range(len(self.x_cordinates)): 
          self.x_values[key].append(self.sourceData.getParam(self.x_cordinates[i].split('|')[1],key)[self.x_cordinates[i].split('|')[2]])
        if self.y_cordinates:
          for i in range(len(self.y_cordinates)): self.y_values[key].append(self.sourceData.getParam(self.y_cordinates[i].split('|')[1],key)[self.y_cordinates[i].split('|')[2]])
        if self.z_cordinates:
          for i in range(len(self.z_cordinates)): self.z_values[key].append(self.sourceData.getParam(self.z_cordinates[i].split('|')[1],key)[self.z_cordinates[i].split('|')[2]])
  
  def __executeActions(self):
    if self.dim < 3:
      if 'figure_properties' in self.options.keys():
        key = 'figure_properties'
        if 'figsize' not in self.options[key].keys():   self.options[key]['figsize'  ] = 'None' 
        if 'dpi' not in self.options[key].keys():       self.options[key]['dpi'      ] = 'None'
        if 'facecolor' not in self.options[key].keys(): self.options[key]['facecolor'] = 'None'
        if 'edgecolor' not in self.options[key].keys(): self.options[key]['edgecolor'] = 'None'
        if 'frameon' not in self.options[key].keys():   self.options[key]['frameon'  ] = 'True'
        elif self.options[key]['frameon'].lower() in ['t','true']: self.options[key]['frameon'] = 'True'
        elif self.options[key]['frameon'].lower() in ['f','false']: self.options[key]['frameon'] = 'False'           
        if 'attributes' in self.options[key].keys(): self.plt.figure(num=None, figsize=ast.literal_eval(self.options[key]['figsize']), dpi=ast.literal_eval(self.options[key]['dpi']), facecolor=self.options[key]['facecolor'],edgecolor=self.options[key]['edgecolor'],frameon=ast.literal_eval(self.options[key]['frameon']),**self.options[key]['attrobutes'])
        else: self.plt.figure(num=None, figsize=ast.literal_eval(self.options[key]['figsize']), dpi=ast.literal_eval(self.options[key]['dpi']), facecolor=self.options[key]['facecolor'],edgecolor=self.options[key]['edgecolor'],frameon=ast.literal_eval(self.options[key]['frameon']))
      if 'title' not in self.options.keys(): self.plt.title(self.name)
      for key in self.options.keys():
        if key == 'range': 
          if 'ymin' in self.options[key].keys(): self.plt.ylim(ymin = ast.literal_eval(self.options[key]['ymin']))
          if 'ymax' in self.options[key].keys(): self.plt.ylim(ymax = ast.literal_eval(self.options[key]['ymax']))
          if 'xmin' in self.options[key].keys(): self.plt.xlim(xmin = ast.literal_eval(self.options[key]['xmin']))
          if 'xmax' in self.options[key].keys(): self.plt.xlim(xmax = ast.literal_eval(self.options[key]['xmax']))
        elif key == 'title':
          if 'attributes' in self.options[key].keys(): self.plt.title(self.options[key]['text'],**self.options[key]['attributes'])
          else: self.plt.title(self.options[key]['text'])    
        elif key == 'figure_properties': pass
        elif key == 'add_text':
          if 'position' not in self.options[key].keys(): self.options[key]['position'] = str((min(self.x_values) + max(self.x_values))*0.5) + ',' + str((min(self.y_values) + max(self.y_values))*0.5)  
          if 'fontdict' not in self.options[key].keys(): self.options[key]['fontdict'] = None
          else: 
            try: self.options[key]['fontdict'] = ast.literal_eval(self.options[key]['fontdict'])
            except AttributeError: raise('STREAM MANAGER: ERROR -> In ' + key +' tag: can not convert the string "' + self.options[key]['fontdict'] + '" to a dictionary! Check syntax for python function ast.literal_eval')
          if 'attributes' in self.options[key].keys(): self.plt.text(float(self.options[key]['position'].split(',')[0]),float(self.options[key]['position'].split(',')[1]),self.options[key]['text'],fontdict=self.options[key]['fontdict'],**self.options[key]['attributes'])
          else: self.plt.text(ast.literal_eval(self.options[key]['position'].split(',')[0]),ast.literal_eval(self.options[key]['position'].split(',')[1]),self.options[key]['text'],fontdict=self.options[key]['fontdict'])    
        elif key == 'autoscale':
          if 'enable' not in self.options[key].keys(): self.options[key]['enable'] = 'True'
          elif self.options[key]['enable'].lower() in ['t','true']: self.options[key]['enable'] = 'True'
          elif self.options[key]['enable'].lower() in ['f','false']: self.options[key]['enable'] = 'False' 
          if 'axis' not in self.options[key].keys()  : self.options[key]['axis'] = 'both'
          if 'tight' not in self.options[key].keys() : self.options[key]['tight'] = 'None'
          self.plt.autoscale(enable = ast.literal_eval(self.options[key]['enable']), axis = self.options[key]['axis'], tight = ast.literal_eval(self.options[key]['tight']))
        elif key == 'horizontal_line':
          if 'y' not in self.options[key].keys(): self.options[key]['y'] = '0'
          if 'xmin' not in self.options[key].keys()  : self.options[key]['xmin'] = '0'
          if 'xmax' not in self.options[key].keys() : self.options[key]['xmax'] = '1'
          if 'hold' not in self.options[key].keys() : self.options[key]['hold'] = 'None'
          if 'attributes' in self.options[key].keys(): self.plt.axhline(y=ast.literal_eval(self.options[key]['y']), xmin=ast.literal_eval(self.options[key]['xmin']), xmax=ast.literal_eval(self.options[key]['xmax']), hold=ast.literal_eval(self.options[key]['hold']),**self.options[key]['attributes'])
          else: self.plt.axhline(y=ast.literal_eval(self.options[key]['y']), xmin=ast.literal_eval(self.options[key]['xmin']), xmax=ast.literal_eval(self.options[key]['xmax']), hold=ast.literal_eval(self.options[key]['hold']))
        elif key == 'vertical_line':
          if 'x' not in self.options[key].keys(): self.options[key]['x'] = '0'
          if 'ymin' not in self.options[key].keys()  : self.options[key]['ymin'] = '0'
          if 'ymax' not in self.options[key].keys() : self.options[key]['ymax'] = '1'
          if 'hold' not in self.options[key].keys() : self.options[key]['hold'] = 'None'
          if 'attributes' in self.options[key].keys(): self.plt.axhline(x=ast.literal_eval(self.options[key]['x']), ymin=ast.literal_eval(self.options[key]['ymin']), ymax=ast.literal_eval(self.options[key]['ymax']), hold=ast.literal_eval(self.options[key]['hold']),**self.options[key]['attributes'])
          else: self.plt.axvline(x=ast.literal_eval(self.options[key]['x']), ymin=ast.literal_eval(self.options[key]['ymin']), ymax=ast.literal_eval(self.options[key]['ymax']), hold=ast.literal_eval(self.options[key]['hold']))
        elif key == 'horizontal_rectangle':
          if 'ymin' not in self.options[key].keys(): raise('STREAM MANAGER: ERROR -> ymin parameter is needed for function horizontal_rectangle!!')
          if 'ymax' not in self.options[key].keys(): raise('STREAM MANAGER: ERROR -> ymax parameter is needed for function horizontal_rectangle!!')
          if 'xmin' not in self.options[key].keys()  : self.options[key]['xmin'] = '0'
          if 'xmax' not in self.options[key].keys() : self.options[key]['xmax'] = '1'
          if 'attributes' in self.options[key].keys(): self.plt.axhspan(ast.literal_eval(self.options[key]['ymin']),ast.literal_eval(self.options[key]['ymax']), ymin=ast.literal_eval(self.options[key]['xmin']), ymax=ast.literal_eval(self.options[key]['xmax']),**self.options[key]['attributes'])
          else:self.plt.axhspan(ast.literal_eval(self.options[key]['ymin']),ast.literal_eval(self.options[key]['ymax']), xmin=ast.literal_eval(self.options[key]['xmin']), xmax=ast.literal_eval(self.options[key]['xmax']))
        elif key == 'vertical_rectangle':
          if 'xmin' not in self.options[key].keys(): raise('STREAM MANAGER: ERROR -> xmin parameter is needed for function vertical_rectangle!!')
          if 'xmax' not in self.options[key].keys(): raise('STREAM MANAGER: ERROR -> xmax parameter is needed for function vertical_rectangle!!')
          if 'ymin' not in self.options[key].keys()  : self.options[key]['ymin'] = '0'
          if 'ymax' not in self.options[key].keys() : self.options[key]['ymax'] = '1'
          if 'attributes' in self.options[key].keys(): self.plt.axvspan(ast.literal_eval(self.options[key]['xmin']),ast.literal_eval(self.options[key]['xmax']), ymin=ast.literal_eval(self.options[key]['ymin']), ymax=ast.literal_eval(self.options[key]['ymax']),**self.options[key]['attributes'])
          else:self.plt.axvspan(ast.literal_eval(self.options[key]['xmin']),ast.literal_eval(self.options[key]['xmax']), ymin=ast.literal_eval(self.options[key]['ymin']), ymax=ast.literal_eval(self.options[key]['ymax']))
        elif key == 'axes_box': self.plt.box(self.options[key][key])
        elif key == 'axis_properties':
          try:self.plt.axis(ast.literal_eval(self.options[key][key]))
          except: self.plt.axis(self.options[key][key]) 
        elif key == 'grid':
          if 'b' not in self.options[key].keys()  : self.options[key]['b'] = None
          if 'which' not in self.options[key].keys() : self.options[key]['which'] = 'major'
          if 'axis' not in self.options[key].keys() : self.options[key]['axis'] = 'both'
          if 'attributes' in self.options[key].keys(): self.plt.grid(ast.literal_eval(b =self.options[key]['b']),which = ast.literal_eval(self.options[key]['which']), axis=ast.literal_eval(self.options[key]['axis']),**self.options[key]['attributes'])
          else:self.plt.grid(b=(self.options[key]['b']),which = (self.options[key]['which']), axis=(self.options[key]['axis']))
        elif key in ['how','plot_settings']: pass
        else:
          command_args = ''
          for kk in self.options[key]:
            if kk != 'attributes' and kk != key:
              if command_args != '(': prefix = ','
              else: prefix = '' 
              try: command_args = prefix + command_args + kk + '=' + str(ast.literal_eval(self.options[key][kk]))
              except:command_args = prefix + command_args + kk + '="' + str(self.options[key][kk])+'"'  
          exec('self.plt.' + key + '(' + command_args + ')')
    else:
      if 'figure_properties' in self.options.keys():
        key = 'figure_properties'
        if 'figsize' not in self.options[key].keys():   self.options[key]['figsize'  ] = 'None' 
        if 'dpi' not in self.options[key].keys():       self.options[key]['dpi'      ] = 'None'
        if 'facecolor' not in self.options[key].keys(): self.options[key]['facecolor'] = 'None'
        if 'edgecolor' not in self.options[key].keys(): self.options[key]['edgecolor'] = 'None'
        if 'frameon' not in self.options[key].keys():   self.options[key]['frameon'  ] = 'True'
        elif self.options[key]['frameon'].lower() in ['t','true']: self.options[key]['frameon'] = 'True'
        elif self.options[key]['frameon'].lower() in ['f','false']: self.options[key]['frameon'] = 'False'           
        if 'attributes' in self.options[key].keys(): self.plt.figure(num=None, figsize=ast.literal_eval(self.options[key]['figsize']), dpi=ast.literal_eval(self.options[key]['dpi']), facecolor=self.options[key]['facecolor'],edgecolor=self.options[key]['edgecolor'],frameon=ast.literal_eval(self.options[key]['frameon']),**self.options[key]['attrobutes'])
        else: self.plt.figure(num=None, figsize=ast.literal_eval(self.options[key]['figsize']), dpi=ast.literal_eval(self.options[key]['dpi']), facecolor=self.options[key]['facecolor'],edgecolor=self.options[key]['edgecolor'],frameon=ast.literal_eval(self.options[key]['frameon']))
      if 'title' not in self.options.keys(): self.Ax.set_title(self.name)
      for key in self.options.keys():
        if key == 'range': 
          if 'xmin' in self.options[key].keys(): self.Ax.set_xlim3d(xmin = ast.literal_eval(self.options[key]['xmin']))
          if 'xmax' in self.options[key].keys(): self.Ax.set_xlim3d(xmax = ast.literal_eval(self.options[key]['xmax']))
          # Disabled...bug in MATPLOTLIB
          #if 'ymin' in self.options[key].keys(): self.Ax.set_ylim3d(ymin = ast.literal_eval(self.options[key]['ymin']))
          #if 'ymax' in self.options[key].keys(): self.Ax.set_ylim3d(ymax = ast.literal_eval(self.options[key]['ymax']))
          if 'zmin' in self.options[key].keys(): 
            self.Ax.set_zlim(ast.literal_eval(self.options[key]['zmin']),ast.literal_eval(self.options[key]['zmax']))
        elif key== 'scale':
          if 'xscale' in self.options[key].keys(): self.Ax.set_xscale(self.options[key]['xscale'])
          if 'yscale' in self.options[key].keys(): self.Ax.set_yscale(self.options[key]['yscale'])        
          if 'zscale' in self.options[key].keys(): self.Ax.set_zscale(self.options[key]['zscale'])
        elif key == 'title':
          if 'attributes' in self.options[key].keys(): self.Ax.set_title(self.options[key]['text'],**self.options[key]['attributes'])
          else: self.plt3D.set_title(self.options[key]['text'])    
        elif key == 'figure_properties': pass
        elif key == 'add_text':
          if 'position' not in self.options[key].keys(): self.options[key]['position'] = str((min(self.x_values) + max(self.x_values))*0.5) + ',' + str((min(self.y_values) + max(self.y_values))*0.5)  
          if 'fontdict' not in self.options[key].keys(): self.options[key]['fontdict'] = 'None'
          if 'withdash' not in self.options[key].keys(): self.options[key]['withdash'] = 'False' 
          if len(self.options[key]['position'].split(',')) < 3: raise('STREAM MANAGER: ERROR -> in 3D plot add_text needs an x,y,z coordinate input the position!!!')
          if 'attributes' in self.options[key].keys(): self.Ax.text(float(self.options[key]['position'].split(',')[0]),float(self.options[key]['position'].split(',')[1]),float(self.options[key]['position'].split(',')[2]),self.options[key]['text'],fontdict=ast.literal_eval(self.options[key]['fontdict']),withdash=ast.literal_eval(self.options[key]['withdash']),**self.options[key]['attributes'])
          else: self.Ax.text(float(self.options[key]['position'].split(',')[0]),float(self.options[key]['position'].split(',')[1]),float(self.options[key]['position'].split(',')[2]),self.options[key]['text'],fontdict=ast.literal_eval(self.options[key]['fontdict']),withdash=ast.literal_eval(self.options[key]['withdash']))
        elif key in ['vertical_rectangle','vertical_line','horizontal_rectangle','horizontal_line']:pass
        elif key == 'autoscale':
          if 'enable' not in self.options[key].keys(): self.options[key]['enable'] = 'True'
          elif self.options[key]['enable'].lower() in ['t','true']: self.options[key]['enable'] = 'True'
          elif self.options[key]['enable'].lower() in ['f','false']: self.options[key]['enable'] = 'False' 
          if 'axis' not in self.options[key].keys()  : self.options[key]['axis'] = 'both'
          if 'tight' not in self.options[key].keys() : self.options[key]['tight'] = 'None'
          self.Ax.autoscale(enable = ast.literal_eval(self.options[key]['enable']), axis = self.options[key]['axis'], tight = ast.literal_eval(self.options[key]['tight']))
        elif key == 'grid':
          if 'b' not in self.options[key].keys()  : self.options[key]['b'] = 'True'
          if self.options[key]['b'].lower() in ['on','t','true']: self.options[key]['b'] = 'True'
          elif self.options[key]['b'].lower() in ['off','f','false']: self.options[key]['b'] = 'False'
          if 'attributes' in self.options[key].keys(): self.Ax.grid(b=ast.literal_eval(self.options[key]['b']),**self.options[key]['attributes'])
          else:self.Ax.grid(b=ast.literal_eval(self.options[key]['b']))
        elif key in ['how','plot_settings']: pass
        else:
          command_args = ''
          for kk in self.options[key]:
            if kk != 'attributes' and kk != key:
              if command_args != '(': prefix = ','
              else: prefix = '' 
              try: command_args = prefix + command_args + kk + '=' + str(ast.literal_eval(self.options[key][kk]))
              except:command_args = prefix + command_args + kk + '="' + str(self.options[key][kk])+'"'  
          exec('self.Ax.' + key + '(' + command_args + ')')

  def readMoreXML(self,xmlNode):
    OutStreamManager.readMoreXML(self,xmlNode)
    if not 'dim' in xmlNode.attrib.keys(): self.dim = 2
    else: self.dim = int(xmlNode.attrib['dim'])
    exec('import matplotlib as ' + 'mpl_' + self.name)
    exec('self.mpl = mpl_' + self.name)
    if self.dim not in [2,3]: raise('STREAM MANAGER: ERROR -> This Plot interface is able to handle 2D-3D plot only')
    exec('import matplotlib.pyplot as ' + 'plt_' + self.name)
    exec('self.plt = plt_' + self.name)
    self.fig = self.plt.figure()
    if self.dim == 3:
      exec('from mpl_toolkits.mplot3d import Axes3D as ' + 'Ax3D_' + self.name)
      exec('self.plt3D = Ax3D_' + self.name)
      exec('self.Ax = Ax3D_' + self.name)
      self.Ax = self.fig.add_subplot(111, projection='3d')

    for subnode in xmlNode:
      if subnode.tag in ['actions']: self.__readPlotActions(subnode)
      if subnode.tag in ['plot_settings']:
        self.options[subnode.tag] = {}
        for subsub in subnode: self.options[subnode.tag][subsub.tag] = subsub.text 
      if subnode.tag in 'title':
        self.options[subnode.tag] = {}
        for subsub in subnode: self.options[subnode.tag][subsub.tag] = subsub.text
        if 'text'     not in self.options[subnode.tag].keys(): self.options[subnode.tag]['text'    ] = node.attrib['name']
        if 'location' not in self.options[subnode.tag].keys(): self.options[subnode.tag]['location'] = 'center'   
      if subnode.tag == 'figure_properties':
        self.options[subnode.tag] = {}
        for subsub in subnode: self.options[subnode.tag][subsub.tag] = subsub.text         
    self.type = 'OutStreamPlot'
    if not 'plot_settings' in self.options.keys(): raise IOError('STREAM MANAGER: ERROR -> For plot named ' + self.name + ' the plot_settings block IS REQUIRED!!')
  
  def addOutput(self,toLoadFrom):
    '''
    Function to add a new output source
    @ In, toLoadFrom, source object
    @ Out, None 
    '''
    self.plt.ioff()
    self.__fillCoordinatesFromSource()
    if self.dim == 2:
      if 'xlabel' not in self.plotSettings.keys():
        x_label = ''
        for index in range(len(self.x_cordinates)) : x_label = x_label + self.x_cordinates[index].split('|')[-1] + ';'
        self.plt.xlabel(x_label)
      else:
        self.plt.xlabel(self.plotSettings['xlabel'])
      if 'ylabel' not in self.plotSettings.keys():
        if self.y_cordinates:
          y_label = ''
          leg_y = ''
          for index in range(len(self.x_cordinates)) : 
            y_label = y_label + self.y_cordinates[index].split('|')[-1] + ','
          self.plt.ylabel(y_label)
          self.plt.legend(ast.literal_eval(y_label))
      else:
        if self.y_cordinates: self.plt.ylabel(self.plotSettings['ylabel'])
       
      if self.outStreamType == 'scatter':
        if 's' not in self.plotSettings.keys(): self.plotSettings['s'] = '20'
        if 'c' not in self.plotSettings.keys(): self.plotSettings['c'] = 'b'
        if 'marker' not in self.plotSettings.keys(): self.plotSettings['marker'] = 'o'   
        if 'alpha' not in self.plotSettings.keys(): self.plotSettings['alpha']='None'
        if 'linewidths' not in self.plotSettings.keys():  self.plotSettings['linewidths'] = 'None'
        for key in self.x_values.keys():
          for x_index in range(len(self.x_values[key])):
            for y_index in range(len(self.y_values[key])):
              if 'attributes' in self.plotSettings.keys(): self.actPlot = self.plt.scatter(self.x_values[key][x_index],self.y_values[key][y_index],s=ast.literal_eval(self.plotSettings['s']),c=(self.plotSettings['c']),marker=(self.plotSettings['marker']),alpha=ast.literal_eval(self.plotSettings['alpha']),linewidths=ast.literal_eval(self.plotSettings['linewidths']),**self.plotSettings['attributes'])
              else: self.actPlot = self.plt.scatter(self.x_values[key][x_index],self.y_values[key][y_index],s=ast.literal_eval(self.plotSettings['s']),c=(self.plotSettings['c']),marker=(self.plotSettings['marker']),alpha=ast.literal_eval(self.plotSettings['alpha']),linewidths=ast.literal_eval(self.plotSettings['linewidths']))
      elif self.outStreamType == 'line':
        for key in self.x_values.keys():
          for x_index in range(len(self.x_values[key])):
            for y_index in range(len(self.y_values[key])):
              if 'attributes' in self.plotSettings.keys(): self.actPlot = self.plt.plot(self.x_values[key][x_index],self.y_values[key][y_index],**self.plotSettings['attributes'])
              else: self.actPlot = self.plt.plot(self.x_values[key][x_index],self.y_values[key][y_index])
      elif self.outStreamType == 'histogram':
        if 'bins' in self.plotSettings.keys(): self.plotSettings['bins'] = ast.literal_eval(self.plotSettings['bins'])
        else: self.plotSettings['bins'] = 10
        if 'normed' not in self.plotSettings.keys(): self.plotSettings['normed'] = False
        else: self.plotSettings['normed'] = ast.literal_eval(self.plotSettings['normed'])
        if 'weights' not in self.plotSettings.keys(): self.plotSettings['weights'] = None
        else: self.plotSettings['weights'] = ast.literal_eval(self.plotSettings['weights'])
        if 'cumulative' not in self.plotSettings.keys(): self.plotSettings['cumulative'] = False
        else: self.plotSettings['cumulative'] = ast.literal_eval(self.plotSettings['cumulative'])
        if 'histtype' not in self.plotSettings.keys(): self.plotSettings['histtype'] = 'bar'
        if 'align' not in self.plotSettings.keys(): self.plotSettings['align'] = 'mid'
        if 'orientation' not in self.plotSettings.keys(): self.plotSettings['orientation'] = 'vertical'                        
        if 'rwidth' not in self.plotSettings.keys(): self.plotSettings['rwidth'] = None
        else: self.plotSettings['rwidth'] = ast.literal_eval(self.plotSettings['rwidth'])
        if 'log' not in self.plotSettings.keys(): self.plotSettings['log'] = None
        else: self.plotSettings['log'] = ast.literal_eval(self.plotSettings['log'])      
        if 'color' not in self.plotSettings.keys(): self.plotSettings['color'] = None
        else: self.plotSettings['color'] = ast.literal_eval(self.plotSettings['color'])   
        if 'stacked' not in self.plotSettings.keys(): self.plotSettings['stacked'] = None
        else: self.plotSettings['stacked'] = ast.literal_eval(self.plotSettings['stacked'])                 
        for key in self.x_values.keys():
          for x_index in range(len(self.x_values[key])):
            if 'attributes' in self.plotSettings.keys(): self.plt.hist(self.x_values[key][x_index], bins=self.plotSettings['bins'], normed=self.plotSettings['normed'], weights=self.plotSettings['weights'], 
                          cumulative=self.plotSettings['cumulative'], histtype=self.plotSettings['histtype'], align=self.plotSettings['align'], 
                          orientation=self.plotSettings['orientation'], rwidth=self.plotSettings['rwidth'], log=self.plotSettings['log'], 
                          color=self.plotSettings['color'], stacked=self.plotSettings['stacked'], **self.plotSettings['attributes'])
            else: self.plt.hist(x, bins=self.plotSettings['bins'], normed=self.plotSettings['normed'], weights=self.plotSettings['weights'], 
                          cumulative=self.plotSettings['cumulative'], histtype=self.plotSettings['histtype'], align=self.plotSettings['align'], 
                          orientation=self.plotSettings['orientation'], rwidth=self.plotSettings['rwidth'], log=self.plotSettings['log'], 
                          color=self.plotSettings['color'], stacked=self.plotSettings['stacked'])       
      elif self.outStreamType == 'stem':
        if 'linefmt' not in self.plotSettings.keys(): self.plotSettings['linefmt'] = 'b-'
        if 'markerfmt' not in self.plotSettings.keys(): self.plotSettings['markerfmt'] = 'bo'
        if 'basefmt' not in self.plotSettings.keys(): self.plotSettings['basefmt'] = 'r-'
        for key in self.x_values.keys():
          for x_index in range(len(self.x_values[key])):
            for y_index in range(len(self.y_values[key])):
              if 'attributes' in self.plotSettings.keys(): self.actPlot = self.plt.stem(self.x_values[key][x_index],self.y_values[key][y_index],linefmt=self.plotSettings['linefmt'], markerfmt=self.plotSettings['markerfmt'], basefmt=self.plotSettings['linefmt'],**self.plotSettings['attributes'])
              else: self.actPlot = self.plt.stem(self.x_values[key][x_index],self.y_values[key][y_index],linefmt=self.plotSettings['linefmt'], markerfmt=self.plotSettings['markerfmt'], basefmt=self.plotSettings['linefmt'])             
      elif self.outStreamType == 'step':
        if 'where' not in self.plotSettings.keys(): self.plotSettings['where'] = 'mid'
        for key in self.x_values.keys():
          for x_index in range(len(self.x_values[key])):
            for y_index in range(len(self.y_values[key])):
              if 'attributes' in self.plotSettings.keys(): self.actPlot = self.plt.step(self.x_values[key][x_index],self.y_values[key][y_index],where=self.plotSettings['where'],**self.plotSettings['attributes'])
              else: self.actPlot = self.plt.step(self.x_values[key][x_index],self.y_values[key][y_index],where=self.plotSettings['where'])
      elif self.outStreamType == 'polar':
        # in here we assume that the x_cordinates are the theta, and y_coordinates are the r(s)
        for key in self.x_values.keys():
          for x_index in range(len(self.x_values[key])):
            for y_index in range(len(self.y_values[key])):
              if 'attributes' in self.plotSettings.keys(): self.actPlot = self.plt.polar(self.x_values[key][x_index],self.y_values[key][y_index],**self.plotSettings['attributes'])
              else: self.actPlot = self.plt.polar(self.x_values[key][x_index],self.y_values[key][y_index])      
      elif self.outStreamType == 'pseudocolor':
        pass
#       if 'alpha' not in self.plotSettings.keys(): self.plotSettings['alpha'] = None
#       else: self.plotSettings['alpha'] = ast.literal_eval(self.plotSettings['alpha']) 
#       if 'C' not in self.plotSettings.keys(): self.plotSettings['C'] = None
#       else: self.plotSettings['alpha'] = ast.literal_eval(self.plotSettings['alpha'])                   
#       if 'edgecolors' not in self.plotSettings.keys(): self.plotSettings['edgecolors'] = None
#       if 'shading' not in self.plotSettings.keys(): self.plotSettings['shading'] = 'flat'
#       
#       for key in self.x_values.keys():
#         for x_index in range(self.x_values[key]):
#           for y_index in range(self.y_values[key]):
#             if 'attributes' in self.plotSettings.keys(): self.actPlot = self.plt.pcolormesh(self.x_values[key][x_index],self.y_values[key][y_index],**self.plotSettings['attributes'])
#             else: self.actPlot = self.plt.pcolormesh(self.x_values[key][x_index],self.y_values[key][y_index])      
    else:
      #3d
      if 'xlabel' not in self.plotSettings.keys():
        x_label = ''
        for index in range(len(self.x_cordinates)) : x_label = x_label + self.x_cordinates[index].split('|')[-1] + ';'
        self.plt3D.set_xlabel(self.Ax,x_label)
      else:
        self.plt3D.set_xlabel(self.Ax,self.plotSettings['xlabel'])
      if 'ylabel' not in self.plotSettings.keys():
        if self.y_cordinates:
          y_label = ''
          leg_y = ''
          for index in range(len(self.x_cordinates)) : 
            y_label = y_label + self.y_cordinates[index].split('|')[-1] + ','
          self.plt3D.set_ylabel(self.Ax,y_label)
      else:
        if self.y_cordinates: self.plt3D.set_ylabel(self.Ax,self.plotSettings['ylabel'])
      if 'zlabel' not in self.plotSettings.keys():
        if self.z_cordinates:
          z_label = ''
          leg_z = ''
          for index in range(len(self.x_cordinates)) : 
            z_label = z_label + self.z_cordinates[index].split('|')[-1] + ','
          self.plt3D.set_zlabel(self.Ax,z_label)
      else:
        if self.z_cordinates: self.plt3D.ylabel(self.Ax,self.plotSettings['ylabel'])
 
      if self.outStreamType == 'scatter':
        if 's' not in self.plotSettings.keys(): self.plotSettings['s'] = '20'
        if 'c' not in self.plotSettings.keys(): self.plotSettings['c'] = 'b'
        if 'marker' not in self.plotSettings.keys(): self.plotSettings['marker'] = 'o'   
        if 'alpha' not in self.plotSettings.keys(): self.plotSettings['alpha']='None'
        if 'linewidths' not in self.plotSettings.keys():  self.plotSettings['linewidths'] = 'None'
        for key in self.x_values.keys():
          for x_index in range(len(self.x_values[key])):
            for y_index in range(len(self.y_values[key])):
              for z_index in range(len(self.z_values[key])):
                if 'attributes' in self.plotSettings.keys(): self.actPlot = self.plt3D.scatter3D(self.x_values[key][x_index],self.y_values[key][y_index],self.z_values[key][z_index],s=ast.literal_eval(self.plotSettings['s']),c=(self.plotSettings['c']),marker=(self.plotSettings['marker']),alpha=ast.literal_eval(self.plotSettings['alpha']),linewidths=ast.literal_eval(self.plotSettings['linewidths']),**self.plotSettings['attributes'])
                else: self.actPlot = self.plt3D.scatter3D(self.Ax,self.x_values[key][x_index],self.y_values[key][y_index],self.z_values[key][z_index],s=ast.literal_eval(self.plotSettings['s']),c=(self.plotSettings['c']),marker=(self.plotSettings['marker']),alpha=ast.literal_eval(self.plotSettings['alpha']),linewidths=ast.literal_eval(self.plotSettings['linewidths']))
      elif self.outStreamType == 'line':
        for key in self.x_values.keys():
          for x_index in range(len(self.x_values[key])):
            for y_index in range(len(self.y_values[key])):
              for z_index in range(len(self.z_values[key])):
                if 'attributes' in self.plotSettings.keys(): self.actPlot = self.plt3D.plot(self.Ax,self.x_values[key][x_index],self.y_values[key][y_index],self.z_values[key][z_index],**self.plotSettings['attributes'])
                else: self.actPlot = self.plt3D.plot(self.Ax,self.x_values[key][x_index],self.y_values[key][y_index],self.z_values[key][z_index])
      elif self.outStreamType == 'surface':
        if 'rstride' not in self.plotSettings.keys(): self.plotSettings['rstride'] = '1'
        if 'cstride' not in self.plotSettings.keys(): self.plotSettings['cstride'] = '1'
        if 'cmap' not in self.plotSettings.keys(): self.plotSettings['cmap'] = 'Accent'
        elif self.plotSettings['cmap'] not in self.mpl.cm.datad.keys(): raise('ERROR. The colorMap you specified does not exist... Available are ' + str(self.mpl.cm.datad.keys()))    
        if 'antialiased' not in self.plotSettings.keys(): self.plotSettings['antialiased']='False'
        if 'linewidth' not in self.plotSettings.keys():  self.plotSettings['linewidth'] = '0'
        if 'interpolation_type' not in self.plotSettings.keys(): self.plotSettings['interpolation_type'] = 'cubic'
        elif self.plotSettings['interpolation_type'] not in ['nearest','linear','cubic']: raise('STREAM MANAGER: ERROR -> surface interpolation unknown. Available are :' + str(['nearest','linear','cubic']))  
        for key in self.x_values.keys():
          for x_index in range(len(self.x_values[key])):
            xi = np.linspace(self.x_values[key][x_index].min(),self.x_values[key][x_index].max(),self.x_values[key][x_index].size)
            for y_index in range(len(self.y_values[key])):
              yi = np.linspace(self.y_values[key][y_index].min(),self.y_values[key][y_index].max(),self.y_values[key][y_index].size)
              xig, yig = np.meshgrid(xi, yi)
              for z_index in range(len(self.z_values[key])):
                if self.plotSettings['interpolation_type'] != 'nearest' and self.z_values[key][z_index].size > 3: zi = griddata((self.x_values[key][x_index],self.y_values[key][y_index]), self.z_values[key][z_index], (xi[:], yi[:]), method=self.plotSettings['interpolation_type'])
                else: zi = griddata((self.x_values[key][x_index],self.y_values[key][y_index]), self.z_values[key][z_index], (xi[:], yi[:]), method='nearest')
                if 'attributes' in self.plotSettings.keys(): self.plt3D.plot_surface(self.Ax,xig,yig,zi, rstride = ast.literal_eval(self.plotSettings['rstride']), cstride=ast.literal_eval(self.plotSettings['cstride']),cmap=self.mpl.cm.get_cmap(name=self.plotSettings['cmap']),linewidth= ast.literal_eval(self.plotSettings['linewidth']),antialiased=ast.literal_eval(self.plotSettings['antialiased']),**self.plotSettings['attributes'])    
                else: self.plt3D.plot_surface(self.Ax,xig,yig,zi,rstride=ast.literal_eval(self.plotSettings['rstride']), cstride=ast.literal_eval(self.plotSettings['cstride']),cmap=self.mpl.cm.get_cmap(name=self.plotSettings['cmap']),linewidth= ast.literal_eval(self.plotSettings['linewidth']),antialiased=ast.literal_eval(self.plotSettings['antialiased'])) 
      elif self.outStreamType == 'tri-surface':
        if 'rstride' not in self.plotSettings.keys(): self.plotSettings['rstride'] = '1'
        if 'cstride' not in self.plotSettings.keys(): self.plotSettings['cstride'] = '1'
        if 'cmap' not in self.plotSettings.keys(): self.plotSettings['cmap'] = 'Accent'
        elif self.plotSettings['cmap'] not in self.mpl.cm.datad.keys(): raise('ERROR. The colorMap you specified does not exist... Available are ' + str(self.mpl.cm.datad.keys()))    
        if 'antialiased' not in self.plotSettings.keys(): self.plotSettings['antialiased']='False'
        if 'linewidth' not in self.plotSettings.keys():  self.plotSettings['linewidth'] = '0'
        if 'interpolation_type' not in self.plotSettings.keys(): self.plotSettings['interpolation_type'] = 'cubic'
        elif self.plotSettings['interpolation_type'] not in ['nearest','linear','cubic']: raise('STREAM MANAGER: ERROR -> surface interpolation unknown. Available are :' + str(['nearest','linear','cubic']))  
        for key in self.x_values.keys():
          for x_index in range(len(self.x_values[key])):
            xi = np.linspace(self.x_values[key][x_index].min(),self.x_values[key][x_index].max(),self.x_values[key][x_index].size)
            for y_index in range(len(self.y_values[key])):
              yi = np.linspace(self.y_values[key][y_index].min(),self.y_values[key][y_index].max(),self.y_values[key][y_index].size)
              xig, yig = np.meshgrid(xi, yi)
              for z_index in range(len(self.z_values[key])):
                if self.plotSettings['interpolation_type'] != 'nearest' and self.z_values[key][z_index].size > 3: zi = griddata((self.x_values[key][x_index],self.y_values[key][y_index]), self.z_values[key][z_index], (xi[:], yi[:]), method=self.plotSettings['interpolation_type'])
                else: zi = griddata((self.x_values[key][x_index],self.y_values[key][y_index]), self.z_values[key][z_index], (xi[:], yi[:]), method='nearest')
                if 'attributes' in self.plotSettings.keys(): self.plt3D.plot_surface(self.Ax,xig,yig,zi, rstride = ast.literal_eval(self.plotSettings['rstride']), cstride=ast.literal_eval(self.plotSettings['cstride']),cmap=self.mpl.cm.get_cmap(name=self.plotSettings['cmap']),linewidth= ast.literal_eval(self.plotSettings['linewidth']),antialiased=ast.literal_eval(self.plotSettings['antialiased']),**self.plotSettings['attributes'])    
                else: self.plt3D.plot_surface(self.Ax,xig,yig,zi,rstride=ast.literal_eval(self.plotSettings['rstride']), cstride=ast.literal_eval(self.plotSettings['cstride']),cmap=self.mpl.cm.get_cmap(name=self.plotSettings['cmap']),linewidth= ast.literal_eval(self.plotSettings['linewidth']),antialiased=ast.literal_eval(self.plotSettings['antialiased'])) 
            
      
      elif self.outStreamType == 'contour':
        pass
      elif self.outStreamType == 'histogram':
        pass
      elif self.outStreamType == 'pseudocolor':
        pass      
      
    if self.interactive: self.plt.ion()
    if 'screen' in self.options['how']['how'].split(','): 
 #     self.plt3D.draw(self.Ax)
      self.plt.show()
    for i in range(len(self.options['how']['how'].split(','))):
      if self.options['how']['how'].split(',')[i].lower() != 'screen':
        self.Ax.savefig(self.name+'_' + self.outStreamType+'.'+self.options['how']['how'].split(',')[i], format=self.options['how']['how'].split(',')[i])        
    self.plt.ioff()
class OutStreamPrint(OutStreamManager):
  def __init(self):
    self.availableOutStreamTypes = ['csv']
  def readMoreXML(self,xmlNode):
    self.type = 'OutStreamPrint'

'''
 Interface Dictionary (factory) (private)
'''
__base                    = 'OutStreamManager'
__interFaceDict           = {}
__interFaceDict['Plot'  ] = OutStreamPlot
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
  



