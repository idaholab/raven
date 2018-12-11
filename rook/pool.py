
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)


import threading
try:
  import queue
except ImportError:
  import Queue as queue
import time

class RunnerThread(threading.Thread):
  """
  This class runs functions in the input queue and puts the results in
  the output queue
  """

  def __init__(self, input_queue, output_queue):
    """
    Initializes with an input queue and an output queue
    Functions and ids in the input_queue will be run and the output
    put into the output queue
    """
    self.__input_queue = input_queue
    self.__output_queue = output_queue
    self.__done = False
    threading.Thread.__init__(self)

  def run(self):
    """
    Runs the functions until the queue is empty.
    """
    try:
      #Keep going as long as there are items in the queue
      while True:
        id_num, function, data = self.__input_queue.get(block=False)
        output = function(data)
        self.__output_queue.put((id_num, output))
        self.__input_queue.task_done()
    except queue.Empty:
      self.__done = True
      return

  def is_done(self):
    """
    Returns true if this is done running.
    """
    return self.__done

class MultiRun:
  """
  This creates queues and runner threads to process the functions.
  """
  def __init__(self, function_list, number_jobs, ready_to_run = None):
    """
    Initializes the class
    function_list: list of functions and data to run
    number_jobs: number of functions to run simultaneously.
    ready_to_run: list of if functions are ready to run
    """
    self.__function_list = function_list
    self.__runners = [None]*number_jobs
    self.__input_queue = queue.Queue()
    self.__output_queue = queue.Queue()
    if ready_to_run is not None:
      assert len(ready_to_run) == len(self.__function_list)
      self.__ready_to_run = ready_to_run[:]
    else:
      self.__ready_to_run = [True]*len(self.__function_list)
    self.__not_ready = 0


  def run(self):
    """
    Starts running all the tests
    """
    self.__not_ready = 0
    for id_num, (function, data) in enumerate(self.__function_list):
      if self.__ready_to_run[id_num]:
        self.__input_queue.put((id_num, function, data))
      else:
        self.__not_ready += 1
    for i in range(len(self.__runners)):
      self.__runners[i] = RunnerThread(self.__input_queue, self.__output_queue)
    for runner in self.__runners:
      runner.start()

  def enable_job(self, id_num):
    """
    Enables the previously not ready job
    """
    assert not self.__ready_to_run[id_num]
    self.__not_ready -= 1
    self.__ready_to_run[id_num] = True
    function, data = self.__function_list[id_num]
    self.__input_queue.put((id_num, function, data))
    for i in range(len(self.__runners)):
      if self.__runners[i].is_done():
        #Restart since it is done.  Otherwise might not be any runners still
        # running
        self.__runners[i] = RunnerThread(self.__input_queue, self.__output_queue)
  def __runner_count(self):
    """
    Returns how many runners are not done.
    """
    runner_count = 0
    for i in range(len(self.__runners)):
      if self.__runners[i].is_alive():
        runner_count += 1
    return runner_count

  def process_results(self, process_function = None):
    """
    Process results and return the output in an array.
    If a process_function is passed in, it will be called with
    process_function(index, input, output) after the output is created.
    """
    return_array = [None]*len(self.__function_list)
    output_count = 0
    count_down = 10
    while output_count < len(return_array):
      while output_count + self.__not_ready < len(return_array):
        #print("meditation numbers oc", output_count, "nr", self.__not_ready, "lra", len(return_array), "rc", self.__runner_count(), "ie", self.__input_queue.empty(), "oe", self.__output_queue.empty())
        id_num, output = self.__output_queue.get()
        count_down = 10
        return_array[id_num] = output
        output_count += 1
        if process_function is not None:
          _, data = self.__function_list[id_num]
          process_function(id_num, data, output)
      time.sleep(0.1)
      if count_down <= 0:
        #Check and see if any runners are still running
        runner_count = self.__runner_count()
        if runner_count == 0:
          break
      count_down -= 1
    assert self.__output_queue.empty(),"Output queue not empty"
    return return_array

  def wait(self):
    """
    wait for all the tasks to be finished
    """
    self.__input_queue.join()
