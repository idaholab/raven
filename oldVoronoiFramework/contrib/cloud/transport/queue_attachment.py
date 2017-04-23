'''
Queue message attachment logic
'''

"""
Copyright (c) 2013 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

email: contact@picloud.com

The cloud package is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this package; if not, see 
http://www.gnu.org/licenses/lgpl-2.1.html    
"""

import numbers
import os
import time
import threading
import thread
import sys
import traceback
from subprocess import Popen, PIPE

from ..cloud import CloudException
from ..queue import Retry, MaxRetryException
from inspect import getargspec

import logging
cloudLog = logging.getLogger('Cloud.queue')


def _handle_action_dct_excp(action_dct, excp, message, cur_retry, max_retries):
    """Push message (and relevant errors) to queue defined by action_dct
    Interprets Retry based on inner_excp
    """
    
    if isinstance(excp, Retry) and MaxRetryException not in action_dct:
        tb = excp.traceback_info
        test_excp = excp.inner_excp_value or excp
    else:
        tb = traceback.format_exc()
        test_excp = excp

    try:
        for parent_excp in test_excp.__class__.mro():
            if parent_excp in action_dct:
                action = action_dct[parent_excp]
    
                error_message = {'exception': excp,
                                 'traceback': traceback.format_exc(),
                                 'source_message': message,
                                 'cur_retry' : cur_retry}
    
                # TODO: Asynchronous?
                action['queue'].push([error_message], delay=action.get('delay', 0))
                break
    except Exception as e:
        print >> sys.stderr, 'Exception handler raised exception'
        traceback.print_exc()
        print >> sys.stderr, ''

def launch_attachment(input_queue, message_handler, expand_iterable_output, output_queues, 
                       batch_size, readers_per_job, retry_on=[], max_retries=None, retry_delay=None,
                       on_error={}, _job_shutdown_timeout=20):
    """Creates the function to be executed on PiCloud."""

    def attachment_runner(message_queue, result_queue):

        message_handler_func = message_handler if callable(message_handler) else message_handler.message_handler
        pre_handling_func = message_handler.pre_handling if callable(getattr(message_handler, 'pre_handling', None)) else None
        post_handling_func = message_handler.post_handling if callable(getattr(message_handler, 'post_handling', None)) else None
        sentinel = None
        
        message_handler_argspec = getargspec(message_handler_func)
        
        if pre_handling_func:
            try:
                pre_handling_func()
            except:
                sys.stderr.write('pre_handling raised exception.\n')
                traceback.print_exc()
                sys.stderr.write('\n')
                sentinel = Exception('pre_handling errored:\n' + traceback.format_exc())

        while True:
            
            encoded_message = message_queue.get()
            #print 'processing %s' % encoded_message
            
            if not encoded_message: # sentinel
                if post_handling_func:
                    try:
                        post_handling_func()
                    except:
                        sys.stderr.write('post_handling raised exception.\n')
                        traceback.print_exc()
                        sys.stderr.write('\n')
                        if not sentinel:
                            sentinel = Exception('post_handling errored:\n' + traceback.format_exc())
                
                result_queue.put({'type' : 'sentinel', 'data' : sentinel})
                break
            
            mid = encoded_message['mid']
            cur_retry = encoded_message.get('cur_retry', 0)
            msg_max_retries = encoded_message.get('max_retries')
            if not msg_max_retries:
                msg_max_retries = max_retries
            
            try:
                
                # decodes can crash from import errors
                message, redirect_key = input_queue._decode_message(encoded_message)
                
                try:
                    # construct what we can pass in
                    kwargs = {}
                    if message_handler_argspec.keywords or 'cur_retry' in message_handler_argspec.args:
                        kwargs['cur_retry'] = cur_retry 
                    if message_handler_argspec.keywords or 'max_retries' in message_handler_argspec.args:
                        kwargs['max_retries'] = max_retries
                    
                    results = message_handler_func(message, **kwargs)
                    
                    if not output_queues: # optimization
                        results = None
                    
                    if not expand_iterable_output:
                        # Returning None indicates no message should be output
                        results = [results] if results != None else []
                        
                    # dev note: results may be a user-defined iterator that can raise exceptions
                    for result in results:
                        
                        # each queue uses a unique bucket reference, so must encode multiple times
                        # TODO: This could be optimized to use a single bucket reference for all output queues                    
                        encoded_results = [output_queue._encode_message(result) 
                                           for output_queue in output_queues]                
                        result_queue.put({'type' : 'encoded_result', 'data' : encoded_results})
                        
                
                except Retry as r: # don't intercept with retry_on
                    raise
                except Exception, e: # automatically raise Retries
                    for test_excp in retry_on:
                        if isinstance(e, test_excp):
                            raise Retry(None,None) 
                    
                    raise
                                    
            except Retry as r:
                
                if r.max_retries is not None:
                    msg_max_retries = r.max_retries
                                    
                effective_delay = r.delay
                if effective_delay is None:
                    effective_delay = retry_delay 
                
                if cur_retry == msg_max_retries:

                    # dynamic replace to a MaxRetryException
                    r.__class__ = MaxRetryException

                    print >> sys.stderr, 'Maximum retries reached'
                    traceback.print_exc()
                    print >> sys.stderr, ''
                    if not sentinel:
                        sentinel = Exception('message_handler retries exceeded:\n' + traceback.format_exc())
                    _handle_action_dct_excp(on_error,r,message, cur_retry, msg_max_retries)
                else:                    
                    cloudLog.warning('Retrying on message. current_retries: %d', cur_retry)                    
                    input_queue._push([message], effective_delay, cur_retry+1, msg_max_retries)                
            
            except Exception as e:
                print >> sys.stderr, 'Attached function raised exception. Message ignored.'
                traceback.print_exc()
                print >> sys.stderr, ''
                if not sentinel:
                    sentinel = Exception('message_handler errored:\n' + traceback.format_exc())
                _handle_action_dct_excp(on_error,e,message, cur_retry, msg_max_retries)
                    
                
            # send done message after putting other things on queue
            result_queue.put({'type' : 'ack', 'data' : {'mid' : mid, 'redirect_key' : redirect_key}})       
    
    def attachment():
        """Responsible for creating subprocesses, each of which will have
        run a reader that processes the input queue."""
        
        # Implemented with multiprocessing queue for very high throughput
        # Downside is that we can have one message always buffered
        
        from multiprocessing import Process, Queue as MPQueue
        from multiprocessing.queues import Empty as QueueEmpty
        from multiprocessing.queues import Full as QueueFull
        
        # Beware of setting _update_deadline too close to _deadline
        # Numbers must be set for worst case of queue saturation
        #    (1 long message, then all fast ones after)
        # As acknowledging is ~4x as fast as popping, need:
        #    _update_deadline / (_deadline - _update_deadline) < 4 
         
        _deadline = 60 
        _update_deadline = 30 # processing time when need to update ack 
        
        # inbound queue workers read from
        # max 1 message so main thread waits until retrieving more messages
        message_queue = MPQueue(1) 
        result_queue = MPQueue(1000) # queue where results are written to
        
        input_queue.raise_exceptions = False # in attachment; continue on errors        
        
        all_processes = []
        for _ in xrange(readers_per_job):
            p = Process(target=attachment_runner, args=(message_queue, result_queue))
            p.start()
            all_processes.append(p)
            
        exception = [None] # list of one element so push_to_queue can modify it
        
        processing_tickets = {} # id to ticket_ids, timeout
        finished_tickets = {} # id to ticket_ids
        ticket_lock = threading.Lock()
        ack_cv = threading.Condition()
        process_monitor_cv = threading.Condition()
        
        shutting_down = [False]
                    
        def push_to_queue():
            """Run indefinitely, popping lists of message responses off mp result queue,
            and pushing them to the output cloud queues.
            
            A message response can be a variety of things:
                
                list: Messages to push to queue
                integer: message id: Ack this
                Exception: error sentinel - mark job errored w/ args[0]
                None: Done sentinel - mark job done
                
            """
            
            max_items = 800 # maximum number of items to push to a queue in a single request
            
            sentinel_signals = 0
            
            while sentinel_signals < readers_per_job:
                                
                data_list = [result_queue.get()]
                dequeue_cnt = 1
                
                # flush queue for batching
                try:
                    while dequeue_cnt < max_items:
                        data_list.append(result_queue.get_nowait())
                        dequeue_cnt+=1 
                except QueueEmpty:
                    pass
                
                # extract types
                sentinels = []
                encoded_message_list = []
                ack_dcts = []
                
                type_def = {'sentinel' : sentinels,
                            'encoded_result' : encoded_message_list,
                            'ack' : ack_dcts}
                
                for data_dct in data_list:
                    type_def[data_dct['type']].append(data_dct['data'])
                    
                # sentinels
                sentinel_signals += len(sentinels)
                if not exception[0]:
                    sentinel_errors = [sentinel for sentinel in sentinels if sentinel]
                    if len(sentinel_errors) > 0:
                        exception[0] = sentinel_errors[0]
                                    
                # An encoded_message has 1 to 1 mapping to output queue it is encoded for
                #  If confused what is going on here, ask Aaron
                for output_queue, encoded_message in zip(output_queues, zip(*encoded_message_list)):
                    output_queue._low_level_push( encoded_message )                    

                
                # now that output has been pushed, it is possible to safely acknowledge
                # for performance reasons, the actual ack is done in a different thread
                with ticket_lock:
                    for ack_dct in ack_dcts:
                        ticket_id = ack_dct['mid']                        
                        try:
                            ticket, _ = processing_tickets.pop(ticket_id)
                        except KeyError: # race condition with monitor_pool
                            if not shutting_down[0]:
                                # FIXME: Should initiate shutdown
                                print >> sys.stderr, 'Error: Could not find ticket %s' % ticket_id
                        else:
                            ack_dct['ticket'] = ticket
                            finished_tickets[ticket_id] = ack_dct
                        
                if ack_dcts:
                    with ack_cv: 
                        ack_cv.notify()
                        
                    
        def ack_handler():
            """Submit acknowledgements and deadline updates for processing jobs"""
            while not shutting_down[0] or finished_tickets or processing_tickets:
                
                # when messages rapidly processing, sleep to allow some more acks to come in
                time.sleep(0.5)
                tickets_to_ack = []
                redirect_keys = []
                 
                with ticket_lock:
                    for ticket_dct in finished_tickets.itervalues():
                        tickets_to_ack.append(ticket_dct['ticket'])
                        redirect_key = ticket_dct['redirect_key']
                        if redirect_key:
                            redirect_keys.append(redirect_key)
                    finished_tickets.clear()
                input_queue.ack(tickets_to_ack)
                
                # handle necessary ack updates                
                now = time.time()
                deadline_update_threshold = now + _update_deadline                
                new_timeout = now + _deadline # when ack_handler will wake again
                 
                need_update_tickets = [] 
                with ticket_lock:
                    for mid, (ticket, timeout) in processing_tickets.items()[:]:
                        if timeout < deadline_update_threshold:
                            need_update_tickets.append(ticket)
                            processing_tickets[mid] = (ticket, now + _deadline - 10)
                        else:
                            new_timeout = min(new_timeout,timeout)
                
                if need_update_tickets:
                    input_queue.update_deadline(need_update_tickets, _deadline)
                    
                # Delete bucket references
                # As this supports 1,000 removes per request, it should be fast enough
                try:
                    input_queue._clean_message(redirect_keys)
                except Exception:
                    logging.warning('Could not remove %s' % redirect_keys, exc_info=True)
                        
                
                if not finished_tickets:
                    with ack_cv: # wait until next message needs ack timeout updated
                        ack_cv.wait(min(0.5,time.time() - new_timeout)) 
                            
        def monitor_pool():
            """Monitor queue pool for unexpected terminations.
            As it is difficult to cleanly handle terminations and they are very rare,
                we simply kill the job if this happens.  
                
            FIXME: Very ugly code. This constantly checks if shutting_down[0] == 2 (system shut down);
                aborts once it is
            """        
            while not shutting_down[0]:
    
                with process_monitor_cv:
                    process_monitor_cv.wait(1)
                
                if shutting_down[0]:                        
                    break             
                
                for p in all_processes:
                    if not p.is_alive():
                        wait_time = 22                                                
                        print >> sys.stderr, 'Subprocess unexpectedly terminated. Giving jobs %ds to finish' % wait_time
                        
                        excp = Exception('Subprocess unexpectedly terminated')
                        result_queue.put({'type' : 'sentinel', 'data' : excp},timeout=1)
                        
                        shutting_down[0] = True
                        with process_monitor_cv:
                            # wait for shut down acknowledgement
                            process_monitor_cv.wait(wait_time)
                            
                            # wait for system exit
                            print >> sys.stderr, 'Wait on system exit'
                            process_monitor_cv.wait(wait_time)

                        if shutting_down[0] == 2:
                            print >> sys.stderr, 'shutdown successful'
                            return
                        else:
                            print >> sys.stderr, 'shutdown not yet successful; terminating job'

                        
                        # empty message queue to get main thread moving again 
                        while True:
                            try:
                                message_queue.get(timeout=1)                                
                            except QueueEmpty:
                                break
                        
                        # take down result queue thread
                        try:
                            for _ in xrange(readers_per_job):                                
                                result_queue.put({'type' : 'sentinel', 'data' : excp},timeout=1)
                            
                            # take down acknowledgment handler
                            with ticket_lock:
                                processing_tickets.clear() # "lost" messages will restart elsewhere   
                            
                            cloudLog.info('monitor pool shutting down')
                            break
                        finally: # the BOMG                                                        
                            with process_monitor_cv:
                                process_monitor_cv.wait(wait_time)
                            if shutting_down[0] == 2:
                                return
                            cloudLog.warning('Safe shutdown could not complete. Hard terminating!')
                            for x in all_processes:
                                x.terminate()
                            time.sleep(5.0)
                            
                            if shutting_down[0] == 2:
                                return                            
                            thread.interrupt_main()
                            time.sleep(2.0)
                            
                            if shutting_down[0] == 2:
                                return                            
                            cloudLog.warning('Monitor aborting process!')
                            os._exit(1)  
                            cloudLog.warning('This line should not have run..')
                        
                    
        push_thread = threading.Thread(target=push_to_queue)
        push_thread.daemon = True
        push_thread.name = 'Output Queue Push Thread'
        push_thread.start()

        ack_thread = threading.Thread(target=ack_handler)
        ack_thread.daemon = True
        ack_thread.name = 'Ack Handler Thread'
        ack_thread.start()

        monitor_thread = threading.Thread(target=monitor_pool)
        monitor_thread.daemon = True
        monitor_thread.name = 'Monitor pool Thread'
        monitor_thread.start()

        cloudLog.debug('All subprocesses started. Listening on queue')
        
        # Main thread processes the input queue
        message_id = 1
        while not shutting_down[0]:
            messages = input_queue._low_level_pop_tickets(batch_size, timeout=_job_shutdown_timeout,
                                                           deadline=_deadline)
            #print 'got messages %s' % messages
            if not messages:  # shut down job
                break 
            
            # inject unique identifiers into message            
            with ticket_lock:
                now = time.time()
                for message in messages:
                    message['mid'] = message_id                    
                    processing_tickets[message_id] = (message['ticket'], now + _deadline - 10)
                    message_id += 1                    
            
            try:
                for message in messages:
                    del message['ticket']
                    message_queue.put(message)
            except QueueFull, qe: # on system exit, this may be raised; translate to a system exit
                cloudLog.info('Got QueueFull (%s) exception. Possibly kill signal?' % str(qe))
                raise SystemExit('Killed')
        
        # SHUTDOWN
        req_shutdown = shutting_down[0]
        if req_shutdown:
            cloudLog.debug('shutdown requested')
        else:
            cloudLog.debug('No more messages in queue -- shutting down')                    
        
        shutting_down[0] = True
        with process_monitor_cv:
            process_monitor_cv.notify()        
        
        # send sentinels
        sentinels_to_send = len(all_processes)
        while sentinels_to_send:
            try:
                message_queue.put(None, timeout=5)
            except QueueFull:
                alive_processes = [p for p in all_processes if p.is_alive()]
                if not alive_processes:
                    break
            else:
                sentinels_to_send-=1

        cloudLog.debug('Waiting for subprocesses to finish')

        # wait for all processes to get sentinel and terminate
        while [p for p in all_processes if p.is_alive()]:
            for p in all_processes:
                p.join(10)
        
        cloudLog.debug('Waiting for outputs to flush.')                    
        push_thread.join()
        
        cloudLog.debug('Waiting for validation to flush.')
        with ack_cv:
            ack_cv.notify() # force wake up
        ack_thread.join()
        
        cloudLog.debug('Done')
        
        shutting_down[0] = 2 # successful shut down
        
        with process_monitor_cv:
            process_monitor_cv.notify()        
        
        if exception[0]:
            raise CloudException(exception[0])
        
    attachment()
