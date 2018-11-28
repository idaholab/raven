
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)


import threading
import queue

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
      return

class MultiRun:
  """
  This creates queues and runner threads to process the functions.
  """
  def __init__(self, function_list, number_jobs):
    """
    Initializes the class
    function_list: list of functions and data to run
    number_jobs: number of functions to run simultaneously.
    """
    self.__function_list = function_list
    self.__runners = [None]*number_jobs
    self.__input_queue = queue.Queue()
    self.__output_queue = queue.Queue()

  def run(self):
    """
    Starts running all the tests
    """
    for id_num, (function, data) in enumerate(self.__function_list):
      self.__input_queue.put((id_num, function, data))
    for i in range(len(self.__runners)):
      self.__runners[i] = RunnerThread(self.__input_queue, self.__output_queue)
    for runner in self.__runners:
      runner.start()


  def process_results(self, process_function = None):
    """
    Process results and return the output in an array.
    If a process_function is passed in, it will be called with
    process_function(index, input, output) after the output is created.
    """
    return_array = [None]*len(self.__function_list)
    output_count = 0
    while output_count < len(return_array):
      id_num, output = self.__output_queue.get()
      return_array[id_num] = output
      output_count += 1
      if process_function is not None:
        _, data = self.__function_list[id_num]
        process_function(id_num, data, output)
    assert self.__output_queue.empty(),"Output queue not empty"
    return return_array

  def wait(self):
    """
    wait for all the tasks to be finished
    """
    self.__input_queue.join()
