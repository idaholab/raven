"""
Queues provide an interface for flow-based programming on PiCloud.

Please see our `Queue documentation <http://docs.picloud.com/queue.html>`_.
"""

from __future__ import with_statement
from __future__ import absolute_import
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

"""Dev beware: list is defined in this module; list will not map to builtin list. Use builtin_list instead"""

import sys
import copy
import json
import time
import uuid
import base64
import string
import types
import numbers

from itertools import islice
from functools import partial
import traceback
import threading
from __builtin__ import list as builtin_list

from functools import wraps
from .serialization import pickledebug as saving_pickle
import cPickle as loading_pickle
from inspect import getargspec

__httpConnection = None
__url = None

from .util import  min_args, max_args

from .util.zip_packer import Packer
from .cloud import CloudException 
from . import _getcloudnetconnection, _getcloud


import logging
cloudLog = logging.getLogger('Cloud.queue')

_queue_create_query = 'queue/'
_queue_list_query = 'queue/'
_queue_count_query = 'queue/{name}/count/'
_queue_exists_query = 'queue/{name}/exists/'
_queue_info_query = 'queue/{name}/'
_queue_delete_query = 'queue/{name}/delete/'
_queue_messages_push_query = 'queue/{name}/push/'
_queue_messages_pop_query = 'queue/{name}/pop/'
_queue_messages_pop_tickets_query = 'queue/{name}/pop_tickets/'
_queue_messages_ack_query = 'queue/{name}/ack/'
_queue_messages_update_deadline_query = 'queue/{name}/update_deadline/'
_queue_attach_query = 'queue/{name}/attach/'
_queue_detach_query = 'queue/{name}/detach/'


def _strip_unicode(s):
    """remove any unicode strings from s"""
    return s.decode('ascii', 'replace').encode('ascii', 'replace')

def _post(conn, url, post_values, headers={}):
    """Use HttpConnection *conn* to issue a post request to *url*
    with values *post_values*. HTTP *headers* may be added as well."""
    
    url = _strip_unicode(url)
    
    if 'success_action_redirect' in headers:
        headers['success_action_redirect'] = _strip_unicode(headers['success_action_redirect'])
    if post_values and 'success_action_redirect' in post_values:
        post_values['success_action_redirect'] = _strip_unicode(post_values['success_action_redirect'])
    
    cloudLog.debug('post url %s with post_values=%s. headers=%s'
                   % (url, post_values, headers))
    response =  conn.post(url, post_values, headers, use_gzip=False)
    
    return response


def list():
    """Returns a list of CloudQueue objects representing all registered
    Queues.
    
    Registration is lazy. See get() for more info."""

    conn = _getcloudnetconnection()
    
    resp = conn.send_request(_queue_list_query, None)

    return [CloudQueue(q) for q in resp['queues']]


def get(name):
    """Returns a CloudQueue object with *name*.
    
    Note that returned CloudQueue is lazily registered on PiCloud. Until an
    operation with the queue is executed, ``exists(name)`` will return False,
    and the queue will not appear in ``list()``"""
    
    return CloudQueue(name)

def exists(name):
    """Returns True if a CloudQueue of *name* is registered."""
    
    conn = _getcloudnetconnection()
    
    resp = conn.send_request(_queue_exists_query.format(name=name), None)
    
    return resp['exists']


class CloudQueue(object):
    """A CloudQueue is a queue registered on PiCloud.
    
    All messages in the queue are stored on PiCloud. Any machine can pop
    these messages, or push more, by creating a CloudQueue with the same
    ``name``."""
    
    # if queue should raise messages on message exceptions
    _raise_message_exceptions = True
    
    # valid letters for the name of a queue 
    _valid_chars_in_name = string.letters + string.digits + '-_'
    
    # maximum number of bytes a message can be (larger messages
    # will be stored in bucket)
    _max_encoded_message_length = 64000

    # maximum number of messages to push in a single push
    _queue_push_limit = 800
    
    # maximum number of seconds to wait for an ack before msg is re-added to queue
    _max_deadline = 43200
    
    # minimum number of seconds
    _min_deadline = 3
    
    
    def __init__(self, name):
        """Creates a new CloudQueue. Use get() instead."""
            
        for c in name:
            if c not in self._valid_chars_in_name:
                raise ValueError('name must only container alphanumeric, '
                                 'underscore, or hyphen characters')
                
        if len(name) >= 40:
            raise ValueError('Length of name (%s) exceeded max of 40 characters'
                              % len(name))
        
        # name of the queue
        self.name = name

    def count(self):
        """Returns an approximation of the number of messages in the queue."""
    
        conn = _getcloudnetconnection()
        
        resp = conn.send_request(_queue_count_query.format(name=self.name), None)
    
        return resp['count']
    
    def info(self):
        """Returns a dictionary describing the queue (notably attachment information)"""
    
        conn = _getcloudnetconnection()
        
        resp = conn.send_request(_queue_info_query.format(name=self.name), None)
    
        return resp['info']
    
    def delete(self):
        """Delete the queue. All messages will be lost."""
    
        conn = _getcloudnetconnection()
        
        resp = conn.send_request(_queue_delete_query.format(name=self.name), None)
    
        return True
    
    def detach(self):
        """Detaches any attached message handler"""
    
        conn = _getcloudnetconnection()
        
        resp = conn.send_request(_queue_detach_query.format(name=self.name), None)
    
        return True
    
    def _encode_message(self, message, cur_retry=0, max_retries=None):
        """Encode a message in our own structure to include metadata.
        
        Our structure uses a top-level dictionary that includes keys for
        the message, which has been serialized, and datatype,which describes
        what serialization method was used."""
        
        json_d = None
        
        base_dictionary = {'cur_retry' : cur_retry,
                           'max_retries' : max_retries} 
        
        try:
            
            # by default, try to json-encode the message
            # NOTE: do we really want dict keys to be converted to strings?
            # this means the message that was pushed won't be identical to the
            # message that is popped.
            serialized_message = json.dumps(message)                
            datatype = 'json'
            json_d = {'message' : serialized_message,
                      'datatype' : datatype}
            
        except (TypeError, UnicodeDecodeError):
            # errors json.dumps may raise
            
            try:
                # fallback if JSON fails -- use pickle
                serialized_message = saving_pickle.dumps(message, 2)
                datatype = 'base64_python_pickle'
                
                # only try to encode it if it is small enough
                if len(serialized_message) < self._max_encoded_message_length:
                    json_d = {'datatype': datatype,
                              'message': base64.b64encode(serialized_message)}
                
            except:
                cloudLog.exception('Could not pickle message')
                raise

        if json_d:
            json_d.update(base_dictionary)
            encoded_message = json.dumps(json_d)
        
        if not json_d or len(encoded_message) > self._max_encoded_message_length:
            from . import bucket
            
            # if the message is too big, store it in user's bucket
            obj_key = 'queue/{name}/{uuid}'.format(name=self.name,
                                                   uuid=uuid.uuid4())
            bucket.putf(serialized_message, obj_key)
            
            if datatype == 'base64_python_pickle':
                datatype = 'python_pickle'
                
            json_d = {'datatype': datatype,
                      'redirect': 'bucket',
                      'key': obj_key}
            json_d.update(base_dictionary)            
            encoded_message = json.dumps(json_d)
        
        return encoded_message        
    
    @classmethod
    def _split_query(cls, query_func, messages):
        """Split messages into chunks of size ``_queue_push_limit``. For
        each chunk, apply *query_func*."""
        
        try:
            messages_iter = iter(messages)
        except TypeError:
            raise TypeError('messages must be an iterable')
                
        while True:
            batched_messages = builtin_list(islice(messages_iter, cls._queue_push_limit))
            if not batched_messages:
                break
            
            query_func(batched_messages)
        
        return True
    
    def push(self, messages, delay=0):
        """Put *messages*, a list of objects, into queue. If *delay*
        is specified, the function will return immediately, but the messages
        will not be available to readers (poppers) for *delay* seconds.
    
        Each object in *messages* must be pickle-able.
        
        Note that if the picked message exceeds 64kb, it will be temporarily saved 
        to your cloud.bucket under the queue/ prefix."""
        
        return self._push(messages,delay)
    
    def _push(self, messages, delay=0, cur_retry=0, max_retries=None):
        """internal push to support propogating retry information"""
        
        if not isinstance(delay, numbers.Integral):
            raise TypeError('delay must be an integer')
        
        if not 900 >= delay >= 0:
            raise ValueError('delay must be between 0 and 900 seconds inclusive.')
        
        if not hasattr(messages, '__iter__'):
            raise ValueError('messages must be an iterable') 

        messages_generator = (self._encode_message(message, cur_retry, max_retries) 
                              for message in messages)
        query_func = partial(self._low_level_push,delay=delay)
                
        return self._split_query(query_func, messages_generator)
    
    def _low_level_push(self, encoded_messages, delay=0):
        """Performs the HTTP request to push encoded messages."""
        conn = _getcloudnetconnection()
        resp = conn.send_request(_queue_messages_push_query.format(name=self.name), 
                         {'message': encoded_messages, 'delay': delay})
        return True
    
    def _decode_message(self, encoded_message):
        """Decodes the message to make it Python-friendly.
        Returns tuple of decoded message and redirect-key (if any) for cleanup purposes
        """
        
        redirect_key = None
        
        # first decode top-level structure
        if isinstance(encoded_message, basestring):
            message_data = json.loads(encoded_message)
        elif isinstance(encoded_message, dict):
            message_data = encoded_message
        else:
            raise TypeError('encoded_message must be a json string or dictionary')
            
        # next, decode message payload
        if isinstance(message_data, dict) and message_data.get('datatype'):
            
            redirect = message_data.get('redirect')
            datatype = message_data.get('datatype')
            
            if redirect and redirect == 'bucket':
                from . import bucket
                redirect_key = message_data['key']
                message_payload = bucket.getf(redirect_key).read()                
            elif redirect:
                # only support redirecting to buckets right now
                raise RuntimeError('Do not understand redirect %s' % redirect)
            else:
                message_payload = message_data['message']
            
            if datatype == 'python_pickle':
                return loading_pickle.loads(message_payload), redirect_key
            elif datatype == 'base64_python_pickle':
                return loading_pickle.loads(base64.b64decode(message_payload)), redirect_key
            elif datatype == 'json':
                return json.loads(message_payload), redirect_key
            else:
                raise Exception('Unknown datatype %s' % datatype)
                        
        else:
            return message_data, redirect_key
        
    def _clean_message(self, redirect_key): 
        """perform necessary clean up after processing a message
        Supports multiple redirect_keys
        """
        if redirect_key:
            from . import bucket
            bucket.remove(redirect_key)    
    
    def _low_level_pop(self, max_count, timeout):
        """Pops a list of json-encoded messages"""
        
        conn = _getcloudnetconnection()
    
        resp = conn.send_request(_queue_messages_pop_query.format(name=self.name), 
                                 {'max_count': max_count, 'timeout': timeout})
    
        return resp['messages']
        
    def _low_level_pop_tickets(self, max_count, timeout, deadline):
        """pops a list of json encoded messages"""
        
        conn = _getcloudnetconnection()
    
        resp = conn.send_request(_queue_messages_pop_tickets_query.format(name=self.name), 
                                 {'max_count': max_count,
                                  'timeout': timeout,
                                  'deadline': deadline})
    
        return resp['messages']


    def _pop_validator(self, max_count, timeout):
        """Checks to ensure arguments to pop are valid."""
        
        if not isinstance(max_count, numbers.Integral):
            raise TypeError('max_count must be an integer')
        
        if not (10 >= max_count > 0):
            raise ValueError('max_count must be between 1 and 10 inclusive')
        
        if not isinstance(timeout, numbers.Integral):
            raise TypeError('timeout must be an integer')
    
        if not (20 >= timeout >= 0):
            raise ValueError('timeout must be between 0 and 20 inclusive')
        
    
    def pop(self, max_count=10, timeout=20):
        """Pops at most *max_count* messages from the queue. Returns a
        list of popped messages. Note that the non-FIFO order of the pop
        is intentional.
        
        Blocks until at least one message is available, or until *timeout*
        seconds elapses, at which point an empty list is returned.
    
        *timeout* must be specified as an integer between 0 and 20 inclusive.
        *max_count* must be specified as an integer between 1 and 10 inclusive."""
    
        self._pop_validator(max_count, timeout)
        
        results_data = self._low_level_pop(max_count, timeout)
        
        messages = []
        
        for datum in results_data:
            try:
                decoded_msg, redirect_key = self._decode_message(datum)
                messages.append(decoded_msg)
            except:
                cloudLog.exception('Could not decode a message')
                if not self._raise_message_exceptions:
                    sys.stderr.write('Could not decode a message.\n')
                    traceback.print_exc()
                    sys.stderr.write('\n')
                else:
                    raise
            else:
                self._clean_message(redirect_key)
                
        return messages

    def pop_tickets(self, max_count=10, timeout=20, deadline=30):
        """Pops at most *max_count* messages from the queue. Returns a list
        of MessageTickets.
        
        Semantics are similar to pop, except that instances of MessageTicket
        are returned which encapsulates the actual message.
            
        All MessageTickets must be acknowledged within *deadline* seconds.
        If they are not the message will be re-added to the queue.
        
        Using pop_tickets and MessageTicket.ack() (or ack_messages) allows
        you to build highly robust messaging systems where a message will
        go unprocessed.""" 
        
        self._pop_validator(max_count, timeout)
        
        if not isinstance(deadline, numbers.Integral):
            raise TypeError('deadline must be an integer')
                
        if not (self._max_deadline >= deadline >= self._min_deadline):
            raise ValueError('deadline must be between %s and %s inclusive'
                             % (self._min_deadline, self._max_deadline))        
        
        
        results_data = self._low_level_pop_tickets(max_count, timeout, deadline)
        
        messages = []        
        
        for decoded_wrapper_dct in results_data:
            try:   
                decoded_message, redirect_key = self._decode_message(decoded_wrapper_dct)                
                ticket = MessageTicket(self, decoded_message, decoded_wrapper_dct['ticket'], 
                                       deadline, redirect_key=redirect_key)
                messages.append(ticket)
            except:
                cloudLog.exception('Could not decode a message')
                if not self._raise_message_exceptions:
                    sys.stderr.write('Could not decode a message.\n')
                    traceback.print_exc()
                    sys.stderr.write('\n')
                else:
                    raise
                
        return messages

    
    def attach(self, message_handler, output_queues=[], iter_output=False, 
                readers_per_job=1, max_parallel_jobs=1, retry_on=[],    
                max_retries=None, retry_delay=None, on_error={}, **kwargs):
        """Register *message_handler* as the processor for this queue on PiCloud.
        
        If ``message_handler`` is a function (or any callable), then it will be
        invoked for each message, with the message as the only argument.
        
        The queue processing will occur in the context of a job. The job automatically
        ends if no data is available in the queue for 20 seconds.
        
        Alternatively, ``message_handler`` may be an instance if it is desired to maintain state 
        between message processing. In this case, the mssage_handler function will be the 
        instance's method "message_handler()" which takes a single argument (a message).        
        The instance may optionally provide ``pre_handling`` and ``post_handling`` methods to manage
        startup and cleanup respectively. ``pre_handling`` is called once before the first 
        message is processed by the wrapping job, and ``post_handling`` is called once after the last
        message is processed by the wrapping job.  
        
        By default, the return value of the message_handler is pushed to each queue in ``output_queues``.
        If the return value is None, no message is pushed.
        
        If ``iter_output`` is set to True, the return value of the message_handler will be considered
        an iterable.  Each element of the iterator (including any None values) will be pushed to 
        each queue in ``output_queues``. 
        
        ``readers_per_job controls`` the number of messages can be processed by a job in parallel.
        
        ``max_parallel_jobs`` is the maximum number of jobs that can run in parallel to process your queue. 
            Set to 0 to disable processing
        
        ``retry_on``, ``max_retries``, ``retry_delay`` and ``on_error`` all relate to error handling.
        Please see our online documentation: http://docs.picloud.com/queue.html#error-handling
        
        Certain special *kwargs* associated with cloud.call can be used for your *message_handler*: 
        
        * _cores:
            Set number of cores your job will utilize. See http://docs.picloud.com/primer.html#choose-a-core-type/
            In addition to having access to more CPU cores, the amount of RAM available will grow linearly.
            Possible values for ``_cores`` depend on what ``_type`` you choose:
            
            * c1: 1                    
            * c2: 1, 2, 4, 8
            * f2: 1, 2, 4, 8, 16                                    
            * m1: 1, 2, 4, 8
            * s1: 1        
        * _env:
            A string specifying a custom environment you wish to run your job within.
            See environments overview at http://docs.picloud.com/environment.html
        * _label: 
            A user-defined string label that is attached to the jobs. Labels can be
            used to filter when viewing jobs interactively (i.e. on the PiCloud website).         
        * _os_env_vars:
            List of operating system environment variables that should be copied to PiCloud from your system
            Alternatively a dictionary mapping the environment variables to the desired values.                                                                 
        * _priority: 
            A positive integer denoting the jobs' priority. PiCloud tries to run jobs 
            with lower priority numbers before jobs with higher priority numbers.                
        * _profile:
            Set this to True to enable profiling of your code. Profiling information is 
            valuable for debugging, and finding bottlenecks.
            **Warning**: Profiling can slow your job significantly; 10x slowdowns are known to occur
        * _restartable:
            In the rare event of hardware failure while a job is processing, this flag
            indicates whether the job can be restarted. By default, this is True. Jobs
            that modify external state (e.g. adding rows to a database) should set this
            False to avoid potentially duplicate modifications to external state.
        * _type:
            Choose the type of core to use, specified as a string:
            
            * c1: 1 compute unit, 300 MB ram, low I/O (default)                    
            * c2: 2.5 compute units, 800 MB ram, medium I/O
            * f2: 5.5 compute units, 3.75 GB ram, high I/O, hyperthreaded core                                    
            * m1: 3.25 compute units, 8 GB ram, high I/O
            * s1: Up to 2 compute units (variable), 300 MB ram, low I/O, 1 IP per core
                           
            See http://www.picloud.com/pricing/ for pricing information
        * _vol:
            A string or list of strings specifying a volume(s) you wish your jobs to have access to. 
                     
        """
        
        # TODO: Bring back batch_size to the public API. If we do, it will
        # represent the number of messages an attachment should get at once.
        
        cloud = _getcloud()
        params = cloud._getJobParameters(message_handler, kwargs, 
                                         ignore=['_depends_on', '_depends_on_errors', 
                                                 '_max_runtime', '_fast_serialization',
                                                 '_kill_process'])
        
        return _low_level_attach(self, message_handler, iter_output, output_queues, None, 
                                 readers_per_job, max_parallel_jobs, retry_on, 
                                 max_retries, retry_delay, on_error, params)


    def __repr__(self):
        return "CloudQueue('{name}')".format(name=self.name)
    
    def __eq__(self, other):
        if not isinstance(other, CloudQueue):
            return False
        return self.name == other.name
    
    def _validate_tickets(self, message_tickets):
        if not hasattr(message_tickets, '__iter__'):
            raise TypeError('message_tickets must be an iterable')
        
        for ticket in message_tickets:
            if isinstance(ticket, MessageTicket):
                if ticket.queue and ticket.queue != self:
                    raise ValueError('MessageTicket is not associated with this queue')
            elif not isinstance(ticket, basestring):
                raise TypeError('message_tickets elements must be MessageTicket or strings')

    
    def ack(self, message_tickets):
        """Ack(nowledge) the ticket - and successful handling of - multiple
        *message_tickets*. If acknowledge is not called, the message will be
        automatically placed back into the queue when deadline is reached.
        
        *message_tickets* can be a MessageTicket or the actual ticket id.
        """ 

        self._validate_tickets(message_tickets)
        
        conn = _getcloudnetconnection()
        
        def query_func(batched_tickets):
            conn.send_request(_queue_messages_ack_query.format(name=self.name), 
                              {'ticket' : batched_tickets})

        ticket_generator = ((mt.ticket if isinstance(mt, MessageTicket) else mt)
                            for mt in message_tickets)
        
        retval = self._split_query(query_func, ticket_generator)
        if retval:
            # clean up
            redirect_generator = ((mt.redirect_key for mt in message_tickets 
                                   if isinstance(mt,MessageTicket) and mt.redirect_key))
            for redirect_key in redirect_generator:
                self._clean_message(redirect_key)
    
    def update_deadline(self, message_tickets, new_deadline):
        """Modify the deadline of multiple not-yet ack-ed *message_tickets*
        After invoking this function, the message_tickets' deadline will 
        occur in *new_deadline* seconds relative to NOW.
        
        *message_tickets* can be a message_ticket or the ticket id.        
        """
        
        self._validate_tickets(message_tickets)
        if not isinstance(new_deadline, numbers.Integral):
            raise TypeError('new_deadline must be an integer')        
        if not (self._max_deadline >= new_deadline >= self._min_deadline):
            raise ValueError('new_deadline must be between 3 and 43200 inclusive')                 

        conn = _getcloudnetconnection()        
        def query_func(batched_tickets):
            conn.send_request(_queue_messages_update_deadline_query.format(name=self.name), 
                              {'ticket' : batched_tickets, 'deadline' : new_deadline})
        
        ticket_generator = ((mt.ticket if isinstance(mt, MessageTicket) else mt)
                            for mt in message_tickets)
        now = time.time()     
        self._split_query(query_func, ticket_generator)
        
        for message in message_tickets:
            if isinstance(message, MessageTicket): 
                message._set_deadline(new_deadline, now)                        
    
class MessageTicket(object):
    """A MessageTicket includes information and functions for
    acknowleding a message, or changing its deadline.
    It also includes the message itself. 
    
    To access the message, read the ``message`` attribute.
    To access ticket id, read the ``ticket_id`` attribute
    """
    
    # queue that message was popped from
    queue = None
    
    # actual message
    message = '' 
    
    # ticket uniquely represents the delivery of the message
    ticket = ''
    
    # number of seconds from receipt of message until this message is
    # automatically re-added to the queue
    deadline = None
    
    # time.time() when object was created (needed for deadline)
    _creation_time = None
    
    # optional redirect_key (external location to be cleaned up when message is acknowledged)
    redirect_key = None
    
    
    def __init__(self, queue, message, ticket, deadline, creation_time=None, redirect_key=None):
        """Creates a new MessageTicket. Do not use directly, use
        ``pop_tickets`` instead."""
    
        self.queue = queue
        self.message = message
        self.ticket = ticket
        self._set_deadline(deadline, creation_time)
        self.redirect_key = redirect_key
        
    def _set_deadline(self, deadline, creation_time):
        """Internal function for modifying the deadline"""
        
        self.deadline = deadline
        self._creation_time = creation_time or time.time()
    
    def ack(self):
        """Acknowledge this ticket. For batch acknowledgements,
        see CloudQueue.acknowledge_message"""
        
        return self.queue.ack([self])
    
    def time_until_deadline(self):
        """Returns how much time (in seconds) is left until the message's
        deadline."""
        return self._creation_time + self.deadline - time.time()
    
    def update_deadline(self, new_deadline):
        """If the deadline you initially set it insufficient (the message cannot
        be reliably purged from the queue), update its deadline. The *new_deadline*
        should be the number of seconds in the future (relative to now) before
        the message is re-added to the queue automatically."""
        
        return self.queue.update_deadline([self], new_deadline)

    def __repr__(self):
        return "MessageTicket('{message}')".format(message=self.message)
    
class Retry(Exception):
    """Raise this Exception within an attachment to abort and retry the message
    
    Within the constructor, the delay muyst be set to specify how many seconds
        to wait until the retried message can be dequeued.
        max_retries must also be set to specify how many times the message can be retried    
    """
    
    # Parameters for retried message itself
    delay = None  # If set, override default delay
    max_retries = None # if set, override default max_retries
        
    
    # information
    inner_excp_value = None # inner exception that triggered this retry 
    traceback_info = None # traceback of the inner exception
    message = '' # some optional extra message
    
    
    def __init__(self, delay, max_retries, message='',
                 inner_excp_value=None, traceback_info=None, **kwargs):
        
        if not isinstance(delay, (numbers.Integral,types.NoneType)):
            raise TypeError('delay must be an integer')

        if not isinstance(max_retries, (numbers.Integral,types.NoneType)):
            raise TypeError('max_retries must be an integer')
                        
        auto_excp_type, auto_excp_value, auto_traceback = sys.exc_info()
        if not traceback_info:        
            traceback_info = ''.join(traceback.format_tb(auto_traceback))
        if not inner_excp_value:
            inner_excp_value = auto_excp_value
        
        self.traceback_info = traceback_info        
        self.inner_excp_value = inner_excp_value
        self.message = message
        self.delay = delay
        self.max_retries = max_retries
        
        super(Retry, self).__init__(delay, max_retries, message, 
                                    inner_excp_value, traceback_info, **kwargs)

    def __str__(self):
        if self.inner_excp_value:            
            return 'Retry(%s) on %s: %s:\n%s' % (self.message, type(self.inner_excp_value), 
                                                                    self.inner_excp_value, 
                                                                    self.traceback_info)
        else:
            return 'Retry(%s)' % self.message
        
    def __repr__(self):
        return self.__str__()    

class MaxRetryException(Retry):
    """Signals maximum retries exceeded"""
    pass


def _validate_action_dct(action_dct):
    """Ensures that an action_dct (on_error) dict is well-formed"""
    
    for e, action in action_dct.items():
        if not issubclass(e, Exception):
            raise ValueError('Key is not an Exception')
        if 'queue' not in action:
            raise ValueError('CloudQueue not specified for action')
        elif not isinstance(action['queue'], CloudQueue):
            raise ValueError('Specified queue must be CloudQueue object')
        if 'delay' in action and not 900 >= action['delay'] >= 0:
            raise ValueError('delay must be between 0 and 900 seconds inclusive.')


def _stringify_action_dct(action_dct):
    """Creates a version of *action_dct* that is composed purely of
    strings. This makes it JSON-able."""
    
    action_dct = copy.deepcopy(action_dct)
    for e, action in action_dct.items():
        action_dct[e.__name__] = action
        del action_dct[e]
        action['queue'] = action['queue'].name
        
    return action_dct

def _low_level_attach(queue, message_handler, expand_iterable_output, output_queues, 
                      batch_size, readers_per_job, max_parallel_jobs, retry_on, 
                      max_retries, retry_delay, on_error, params):
    """Does the grunt work in attaching the *message_handler* to the queue."""

    if not isinstance(readers_per_job, numbers.Integral):
        raise TypeError('readers_per_job must be an integer')
    if not isinstance(max_parallel_jobs, numbers.Integral):
        raise TypeError('timeout must be an integer')
        
    if not (max_parallel_jobs >= 0):
        raise ValueError('max_parallel_jobs must be greater than or equal to 0')
    
    if not (128 >= readers_per_job >= 1):
        raise ValueError('readers_per_job must be between 1 and 128 inclusive')
    
    if not hasattr(retry_on, '__iter__'):
        retry_on = [retry_on]
        
    if retry_on:
        if max_retries is None:
            raise ValueError('max_retries must be set if retry_on is specified')
        if retry_delay is None:
            raise ValueError('retry_delay must be set if retry_on is specified')
        
        if not isinstance(max_retries, numbers.Integral):
            raise TypeError('max_retries must be an integer')
        if not isinstance(retry_delay, numbers.Integral):
            raise TypeError('retry_delay must be an integer')        


    _validate_action_dct(on_error)

    if output_queues:
        if not hasattr(output_queues, '__iter__'):
            output_queues = [output_queues]
        
        if len(output_queues) > 50:
            raise ValueError('Cannot specify more than 50 output queues.')
        
        for output_queue in output_queues:
            if not isinstance(output_queue, CloudQueue):
                raise ValueError('output queues must be CloudQueue instances')

    implements_interface = False

    if not callable(message_handler):
        # if obj has a message_handler member, it's all good
        # if obj has a message_handler member, but not callable, error
        # if obj doesn't have a message_handler member, error with callable needed
        
        if not hasattr(message_handler, 'message_handler'):
            raise TypeError('cloud.queue.attach *message_handler* '
                            'argument (%s) is not callable'
                            % (str(message_handler)))
        
        elif not callable(getattr(message_handler, 'message_handler')):
            raise TypeError('cloud.queue.attach *message_handler* is class without (%s) callable'
                            'message_handler() member function'  % (str(message_handler)))
        
        implements_interface = True
    
    # TODO: batch_size can be determined automatically by measuring the average
    # time and variance it takes to process a message. Assume that a batch pop
    # takes ~100ms, and then do the match.
    # In theory, with more parallel jobs, lower batch size (heroku problem) 
    if not batch_size:
        batch_size = 10  # maximum possible
    
    try:
        if implements_interface:
            min_arg = min_args(message_handler.message_handler)
            max_arg = max_args(message_handler.message_handler)
        else:
            min_arg = min_args(message_handler)
            max_arg = max_args(message_handler)
    except TypeError:
        pass #type can't be sanity checked.. let it through
    else:
        # TODO: Validate special arguments it accepts
        if max_arg < 1:
            raise ValueError('message_handler function must accept 1 (required) argument. %s only accepts %s' \
                             % (str(message_handler), max_arg))
    
    attachment = _create_attachment(queue, message_handler, expand_iterable_output, output_queues, 
                                    batch_size, readers_per_job, retry_on, max_retries, 
                                    retry_delay, on_error)

    cloud = _getcloud()
    conn = _getcloudnetconnection()
    
    os_env_vars = params.pop('os_env_vars', None)
    smessage_handler, sarg, logprefix, logcnt = cloud.adapter.cloud_serialize(attachment, 2, [],
                                                                   logprefix='queue.',
                                                                   os_env_vars=os_env_vars)
    
    conn._update_params(params)
    
    cloud.adapter.dep_snapshot()
    
    data = Packer()
    data.add(smessage_handler)
    params['data'] = data.finish()
    params['output_queue'] = [output_queue.name for output_queue in output_queues]
    params['batch_size'] = batch_size
    params['readers_per_job'] = readers_per_job
    params['max_parallel_jobs'] = max_parallel_jobs
    params['max_retries'] = max_retries
    params['retry_delay'] = retry_delay    
    params['on_error'] = json.dumps(_stringify_action_dct(on_error))
    params['retry_on'] = json.dumps([excp.__name__ for excp in retry_on])
    
    conn.send_request(_queue_attach_query.format(name=queue.name), params) 
     
def _create_attachment(input_queue, message_handler, expand_iterable_output, output_queues, 
                       batch_size, readers_per_job, retry_on=[], max_retries=None, retry_delay=None,
                       on_error={}, _job_shutdown_timeout=20):
    """Creates the function to be executed on PiCloud."""
    
    def attachment():
        from cloud.transport import queue_attachment
        
        return queue_attachment.launch_attachment(input_queue, message_handler, expand_iterable_output, 
                                                  output_queues, batch_size, readers_per_job, 
                                                  retry_on, max_retries, retry_delay, on_error, 
                                                  _job_shutdown_timeout) 
        
    if callable(message_handler):
        return wraps(message_handler)(attachment)
    else:
        return wraps(message_handler.message_handler)(attachment)
