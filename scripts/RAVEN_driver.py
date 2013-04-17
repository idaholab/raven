#!/usr/bin/env python
import subprocess, os, sys, getopt, random, time, shutil
import Queue as queue

pathname = os.path.abspath(os.path.dirname(sys.argv[0]))

class Runner(object):
  def __init__(self,args,input_data=None,input_data_name=None,output_data_name=None):
      self.args = args
      self.input_data = input_data
      self.input_data_name = input_data_name
      self.output_data_name = output_data_name
      self.done = False
      self.start()

  def isDone(self):
    self.process.poll()
    return self.process.returncode != None

  def start(self):
    if self.input_data and self.input_data_name:
      input_file = open(self.input_data_name,"w")
      input_file.write(self.input_data)
      input_file.close()
    if self.output_data_name:
      output_file = open(self.output_data_name,"w")
      stdout = output_file
    else:
      stdout = None
    self.process = subprocess.Popen(self.args,stdout=stdout,stderr=subprocess.STDOUT)

# test = Runner(["cat","foo"],"This is data\nHello World\n","foo","bar")
# test.isDone()

class ProcessQueue(object):
  def __init__(self,num_running=1):
    self.running = [None]*num_running
    self.queue = queue.Queue()
    self.finished = []
    self.next_id = 0
        

  def addItem(self,args,input_data=None,input_data_name=None,output_data_name=None):
    self.queue.put({"args":args,"input_data":input_data,
                        "input_data_name":input_data_name,
                        "output_data_name":output_data_name})
  def isFinished(self):
    self.processEntries()
    if not self.queue.empty():
      return False
    for i in range(len(self.running)):
     if self.running[i] and not self.running[i].isDone():
       return False
    return True
    
  def processEntries(self):
    for i in range(len(self.running)):
      if self.running[i] and self.running[i].isDone():
        self.finished.append(self.running[i])
        self.running[i] = None
    if self.queue.empty():
      return #No more work to do
    for i in range(len(self.running)):
      if self.running[i] == None and not self.queue.empty(): 
        item = self.queue.get()
        args = item["args"]
        for j in range(len(args)):
          arg = args[j]
          arg = arg.replace("%INDEX%",str(i))
          arg = arg.replace("%INDEX1%",str(i+1))
          arg = arg.replace("%CURRENT_ID%",str(self.next_id))
          if item["input_data_name"]:
            arg = arg.replace("%INPUT_FILE%",item["input_data_name"])
            arg = arg.replace("%INPUT_PATH%",os.path.abspath(item["input_data_name"]))
          args[j] = arg
        self.running[i] = Runner(args,item["input_data"],
                                 item["input_data_name"],
                                 item["output_data_name"])
        self.next_id += 1

        


def morgifyInputFile(old_filename,new_filename,morgify_function = None):
  '''
  this function reads an input file (old_filename), create the new input file (new_filename)
  applying the provided change function morgify_function
  '''
  MOOSE_DIR = pathname  + '/../../moose'
  if "MOOSE_DIR" in os.environ:
    MOOSE_DIR = os.environ['MOOSE_DIR']
  elif "MOOSE_DEV" in os.environ:
    MOOSE_DIR = pathname + '/../devel/moose'
    
  sys.path.append(MOOSE_DIR + '/tests')
    
  from ParseGetPot import readInputFile, GPNode
    
  input_data = readInputFile(old_filename)

  def printGpnode(node, depth = 0, output = sys.stdout):
      '''
      recursive function to print all the the YAML nodes
      '''
      indent = "  "
      prefix = indent*max(0,depth - 1)
      if depth == 0:
        pass
      elif depth == 1:
        output.write(prefix+"["+node.name+"]\n")
      else:
        output.write(prefix+"[./"+node.name+"]\n")

      for line in node.comments:
        output.write(prefix + indent + "# " + line+"\n")

      for param_name in node.params_list:
        param_value = "'"+node.params[param_name]+"'"
        param_comment = ""
        if param_name in node.param_comments:
          param_comment = " # "+node.param_comments[param_name]
        output.write(prefix + indent + param_name + " = " + param_value + param_comment+"\n")
        
      for child_name in node.children_list:
        printGpnode(node.children[child_name],depth+1,output)

      if depth == 0:
        pass
      elif depth == 1:
        output.write(prefix+"[]\n")
      else:
        output.write(prefix+"[../]\n")

  if morgify_function:
    morgify_function(input_data)
    
  printGpnode(input_data,output=open(new_filename,"w"))


def createInputChanger(increment,output_base,changerType=None):
  '''
  this function returns the proper input changer described by changerType
  '''
  def input_changerMC(data):
    '''
    this changer alter the seed of the distribution to perform Monte Carlo
    '''
    old_seed = data.children["Distributions"].params.get("RNG_seed",175732028625678) 
    data.children["Distributions"].params["RNG_seed"] = str(int(old_seed) + increment)
  def input_changerFixDistMin(data):
    '''
    this changer force all the distribution to return their lower bound
    '''
    data.children["Distributions"].params["force_distribution"] = 1
   
  def input_changerFixDistAvg(data):
    '''
    this changer force all the distribution to return their average value
    '''
    data.children["Distributions"].params["force_distribution"] = 2
   
  def input_changerFixDistMax(data):
    '''
    this changer force all the distribution to return their upper bound
    '''
    data.children["Distributions"].params["force_distribution"] = 3
      
  def input_changerNone(data):
    '''
    this changer just change the file name
    '''
    data.children["Output"].params["file_base"] = output_base
    return
  def input_changer(data):
    '''
    this create an interface for all changers
    '''
    data.children["Output"].params["file_base"] = output_base
    inputChangersDict = {}
    inputChangersDict['Monte Carlo'] = input_changerMC
    inputChangersDict['FixDistMin' ] = input_changerFixDistMin
    inputChangersDict['FixDistAvg' ] = input_changerFixDistAvg
    inputChangersDict['FixDistMax' ] = input_changerFixDistMax
    inputChangersDict['None' ]       = input_changerNone
    if changerType:
      try:
        inputChangersDict['changerType'](data)
      except:
        raise AttributeError('the requested input file changer is not available')
    else:
      return inputChangersDict['None'](data)
    return

  return input_changer


def runBatches(runs,batch_size,input_filename,input_prefix,base_args,modify_input=None):
  '''
  perform simulation in batches, runs is the total number of simulation,
  batch_size is the maximum size of simulation running simultaneusly,
  input_prefix is the 
  base_args is the string of line commands to be used for the run
  modify_input is a string identifying the type of modification needed 
  '''
  print runs,batch_size,input_filename
  running_queue = ProcessQueue(batch_size)
  counter   = 0
  for i in range(runs):
    new_input = input_prefix+"_"+str(i)
    output_file = input_prefix+"_"+str(i)+".out"
    counter += 1
    if modify_input != 'None':
      morgifyInputFile(input_filename,new_input,createInputChanger(counter,output_file,modify_input))
    else:
      shutil.copyfile(input_filename,new_input)
    
    if len(list(filter(lambda x: "%INPUT_FILE%" in x or "%INPUT_PATH%" in x,base_args))) > 0:
      args = base_args[:]
    else:
      args = base_args+[new_input]
    running_queue.addItem(args,input_data_name=new_input,output_data_name=output_file)
  while not running_queue.isFinished():
    time.sleep(1.0)
    running_queue.processEntries()
  return


#morgifyInputFile("../../tests/core_example/PWR_CoreChannel_pre_dist_test.i","foo.i",create_seed_changer("123456"))

if __name__ == "__main__":
  '''
  main driver for statistical analysis
  '''
  if len(sys.argv) == 1:
    print sys.argv[0],"[--input-file=Input_file_for_the_runner] [-i=Input_file_for_the_runner]" 
    sys.exit(-1)
    opts, args = getopt.getopt(sys.argv[1:],"i:",["input-file="])
    print sys.argv
    raven_runner_dir =  os.path.dirname(os.path.abspath(sys.argv[0]))
    raven_dir = os.path.split(raven_runner_dir)[0]
    








#    runs = 1
#    batch_size = 1
#    input_filename = None
#    input_prefix = "sub"
#    modify_input = True
#    for o,a in opts:
#        if o in ("-b","--batch-size"):
#            batch_size = int(a)
#        elif o in ("-r","--runs"):
#            runs = int(a)
#        elif o in ("-i","--input-file"):
#            input_filename = a
#        elif o in ("-p","--input-prefix"):
#            input_prefix = a
#        elif o in ("-n","--no-modify"):
#            modify_input = False
#        elif o in ("-m","--mode"):
#            mode = a
#            if mode == "pbs":
#                args = ["pbsdsh","-v","-n","%INDEX1","--",
#                        os.path.join(raven_runner_dir,
#                                     "remote_runner.sh"),
#                        "out_%CURRENT_ID%"]+args
#            else:
#                print "UNSUPPORTED Mode",mode
#        else:
#            assert False,"unhandled option"
#    for j in range(len(args)):
#        arg = args[j]
#        arg = arg.replace("%RAVEN_DIR%",raven_dir)
#        args[j] = arg
#    runBatches(runs,batch_size,input_filename,input_prefix,args,modify_input)


#Example run:
#./RAVEN_runner.py --batch-size=3 --runs=10 --input-file=foo_base --input-prefix=foo_run ../../RAVEN-devel -i
