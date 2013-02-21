'''
Created on Feb 6, 2013

@author: crisr
'''
from Datas import setDataType

class MonteCarloSampler:
  '''
  class running a Monte Carlo sampling
  '''
  def __init__(self,name,inputFile,outRoot):
    '''
    Constructor
    '''
    self.name               = name      #name of the instance of the Monte Carlo
    self.outputType         = ''        #define the type of output to keep
    self.n_samples          = 0         #number of sampling to be done
    self.filter             = []        #Type of filter to be applied to the data while recovering them
    self.perturbedParamters = []        #list of char with the name of the parameters perturbed
    self.outputNames        = []        #list of char with the name of the phase space variable recorded
    self.timesnaps          = []        #list of time coordinate for the snaps (if presents)
    self.inputFile          = inputFile #input file to be used for the sampling
    self.outRoot            = outRoot   #output file root
    self.keepFiles          = False     #to delete or keep the files
    self.seekBoundaries     = False

  def run(self):
    '''
    perform all single steps to get the final output
    '''
    self.output = setDataType(self.outputType)(self.name) #generate output container
    if len(self.perturbedParamters)==0:
      self.runSeekPerturbedParamters
    if self.seekBoundaries:
      self.runSeekBoundaries
    run()
    

    
    return

  def setFilter(self,outfilter):
    self.filter = outfilter
  
  def readSetUp(self,data):
    '''
    from the input file reads the information to fill himself
    '''
    def checkNone(inVar,string):
      '''
      check if a variable is initialize a None and return the requested error
      '''
      if inVar == None:
        raise AttributeError(string)
      
    self.outputType         = data.params.get("out_type")
    self.n_samples          = data.params.get("n_samples")
    self.fiter              = data.params.get("filter")
    self.perturbedParamters = data.params.get("inp_params")
    self.outputNames        = data.params.get("outVariables")
    self.keepFiles          = data.params.get("keepFiles")
    self.seekBoundaries     = data.params.get("Boundaries")
    checkNone(self.outputType,'A output type is request')
    checkNone(self.outputType,'Number of sampling is requested')
    if self.outputNames == None:
      self.outputNames[0] = 'all'
    if self.keepFiles == None:
      self.keepFiles = True
    if self.seekBoundaries == None:
      self.seekBoundaries = True

  def getOutput(self):
    '''
    from the output  file reads the information to fill himself
    '''
    self.output.load(self.outputType,self.n_samples,self.perturbedParamters,self.outputNames,self.filter)
      

 