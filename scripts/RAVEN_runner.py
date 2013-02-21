#!/usr/bin/env python
import subprocess, os, sys, getopt, random, time, shutil, signal
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
        
        print(self.args)
#        self.process = subprocess.Popen(self.args)
        self.process = subprocess.Popen(self.args,stdout=stdout,stderr=subprocess.STDOUT)

    def kill(self):
        #In python 2.6 this could be self.process.terminate()
        print "Terminating ",self.process.pid,self.args
        os.kill(self.process.pid,signal.SIGTERM)

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
        
    def terminateAll(self):
        #clear out the queue
        while not self.queue.empty():
            self.queue.get()
        for i in range(len(self.running)):
            self.running[i].kill()

        


def morgifyInputFile(old_filename,new_filename,morgify_function = None):
    MOOSE_DIR = pathname  + '/../../moose'
    if "MOOSE_DIR" in os.environ:
      MOOSE_DIR = os.environ['MOOSE_DIR']
    elif "MOOSE_DEV" in os.environ:
      MOOSE_DIR = pathname + '/../devel/moose'


    sys.path.append(MOOSE_DIR + '/tests')

    from ParseGetPot import readInputFile, GPNode
    
    input_data = readInputFile(old_filename)
    def printGpnode(node, depth = 0, output = sys.stdout):
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


def createInputChanger(increment,output_base):
    def input_changer(data):
        old_seed = data.children["Distributions"].params.get("RNG_seed",175732028625678) 
        data.children["Distributions"].params["RNG_seed"] = str(int(old_seed) + increment)
        data.children["Output"].params["file_base"] = output_base
    return input_changer

def getFibonacci(fibo_1,fibo_2):
    return (fibo_1+fibo_2)

def generateSeed():
    return random.randint(0,2**32)


def runBatches(runs,batch_size,input_filename,input_prefix,base_args,modify_input):
    print runs,batch_size,input_filename
    running_queue = ProcessQueue(batch_size)
    def sigterm_handler(signal,frame):
        running_queue.terminateAll()
        return
    
    signal.signal(signal.SIGTERM,sigterm_handler)
    #initialize fibonacci
    fibo_1 = 0
    #fibo_2 = 1
    increment = 1
    for i in range(runs):
        new_input = input_prefix+"_"+str(i)
        output_file = input_prefix+"_"+str(i)+".out"
        # get fibonacci series number
        #increment = getFibonacci(fibo_1,fibo_2)
        increment = increment + fibo_1
        # re-initialize
        fibo_1 = fibo_1 + 1
        #fibo_1 = fibo_2
        #fibo_2 = increment
        print('i '+str(i))
        if modify_input:
            morgifyInputFile(input_filename,new_input,
                             createInputChanger(increment,output_file))
        else:
            shutil.copyfile(input_filename,new_input)
        if len(list(filter(lambda x: "%INPUT_FILE%" in x or "%INPUT_PATH%" in x,base_args))) > 0:
            args = base_args[:]
        else:
            args = base_args+[new_input]
        running_queue.addItem(args,input_data_name=new_input,output_data_name=output_file)
        print('item added')
    while not running_queue.isFinished():
        time.sleep(1.0)
        running_queue.processEntries()
    return


#morgifyInputFile("../../tests/core_example/PWR_CoreChannel_pre_dist_test.i","foo.i",create_seed_changer("123456"))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print sys.argv[0],"[--batch-size=Number_to_run_at_same_time] [--runs=total_to_run] [--input-file=raven_input_file] [--input-prefix=prefix] [--no-modify] [--mode=pbs] command_to_run" 
            #Arguments --batch-size=4 (Number to run
        sys.exit(-1)
    opts, args = getopt.getopt(sys.argv[1:],"b:r:i:p:nm:",
                               ["batch-size=","runs=","input-file=","input-prefix=","no-modify","mode="])
    print sys.argv
    raven_runner_dir =  os.path.dirname(os.path.abspath(sys.argv[0]))
    raven_dir = os.path.split(raven_runner_dir)[0]
    runs = 1
    batch_size = 1
    input_filename = None
    input_prefix = "sub"
    modify_input = True
    for o,a in opts:
        if o in ("-b","--batch-size"):
            batch_size = int(a)
        elif o in ("-r","--runs"):
            runs = int(a)
        elif o in ("-i","--input-file"):
            input_filename = a
        elif o in ("-p","--input-prefix"):
            input_prefix = a
        elif o in ("-n","--no-modify"):
            modify_input = False
        elif o in ("-m","--mode"):
            mode = a
            if mode == "pbs":
                args = ["pbsdsh","-v","-n","%INDEX1","--",
                        os.path.join(raven_runner_dir,
                                     "remote_runner.sh"),
                        "out_%CURRENT_ID%"]+args
            else:
                print "UNSUPPORTED Mode",mode
        else:
            assert False,"unhandled option"
    for j in range(len(args)):
        arg = args[j]
        arg = arg.replace("%RAVEN_DIR%",raven_dir)
        args[j] = arg
    print('input_filename '+input_filename)
    print('input_prefix '+input_prefix)
    runBatches(runs,batch_size,input_filename,input_prefix,args,modify_input)


#Example run:
#./RAVEN_runner.py --batch-size=3 --runs=10 --input-file=foo_base --input-prefix=foo_run ../../RAVEN-devel -i
