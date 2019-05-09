# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module implements a thread pool for running tasks.
"""
from __future__ import division, print_function, absolute_import
import warnings

import threading
try:
  import queue
except ImportError:
  import Queue as queue
import time

warnings.simplefilter('default', DeprecationWarning)

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
      @ In, input_queueo, queue.Queue, queue with the input data and functions to run
      @ In, output_queue, queue.Queue, queue to put result data
      @ Out, None
    """
    self.__input_queue = input_queue
    self.__output_queue = output_queue
    self.__done = False
    threading.Thread.__init__(self)

  def run(self):
    """
      Runs the functions until the queue is empty.
      @ In, None
      @ Out, None
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
      Returns true if this is done running.  Also should check is_alive to
      find out if it failed to successfully finish.
      @ In, None
      @ Out, __done, boolean, true if this successfully finished running.
    """
    return self.__done

class MultiRun:
  """
  This creates queues and runner threads to process the functions.
  """
  def __init__(self, function_list, number_jobs, ready_to_run=None):
    """
      Initializes the class
      @ In, function_list, list, list of functions and data to run
      @ In, number_jobs, int, number of functions to run simultaneously.
      @ In, ready_to_run, list, optional, list of if functions are ready to run
      @ Out, None
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
      @ In, None
      @ Out, None
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
      @ In, id_num, int, id (index in function_list) to enable
      @ Out, None
    """
    assert not self.__ready_to_run[id_num]
    self.__not_ready -= 1
    self.__ready_to_run[id_num] = True
    function, data = self.__function_list[id_num]
    self.__input_queue.put((id_num, function, data))
    self.__restart_runners()

  def __restart_runners(self):
    """
      Restarts any dead runners
      @ In, None
      @ Out, None
    """
    for i in range(len(self.__runners)):
      if self.__runners[i].is_done() or not self.__runners[i].is_alive():
        #Restart since it is done.  Otherwise might not be any runners still
        # running
        self.__runners[i] = RunnerThread(self.__input_queue, self.__output_queue)
        self.__runners[i].start()

  def __runner_count(self):
    """
      Returns how many runners are not done.
      @ In, None
      @ Out, runner_count, int, alive runners
    """
    runner_count = 0
    for i in range(len(self.__runners)):
      if self.__runners[i].is_alive():
        runner_count += 1
    return runner_count

  def process_results(self, process_function=None):
    """
      Process results and return the output in an array.
      If a process_function is passed in, it will be called with
      process_function(index, input, output) after the output is created.
      @ In, process_function, function, optional, function called after each input finishes.
      @ Out, return_array, list, includes the outputs of the functions.
    """
    return_array = [None]*len(self.__function_list)
    output_count = 0
    count_down = 10
    while output_count < len(return_array):
      while output_count + self.__not_ready < len(return_array):
        #This is debug information meant to help if the test system deadlocks
        # It could be commented out, but probably should not be deleted.
        #print("debug numbers oc", output_count, "nr", self.__not_ready,
        #      "lra", len(return_array), "rc", self.__runner_count(), "ie",
        #      self.__input_queue.empty(), "oe", self.__output_queue.empty())
        if self.__runner_count() == 0 and not self.__input_queue.empty():
          print("restarting runners")
          self.__restart_runners()
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
    assert self.__output_queue.empty(), "Output queue not empty"
    return return_array

  def wait(self):
    """
      wait for all the tasks to be finished
      @ In, None
      @ Out, None
    """
    self.__input_queue.join()
