"""
DEPRECATED: Please use cloud.bucket

For managing files on PiCloud's S3 store.

.. note::

    This module cannot be used to access files stored on a job's mounted file system
"""

# TODO: Support files larger than 5 gb

from __future__ import with_statement
from __future__ import absolute_import
"""
Copyright (c) 2011 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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

"""User beware: list is defined in this module; list will not map to builtin list!"""

import os
import sys
import logging
import time
import socket
import errno
import random
import urllib2

__httpConnection = None
__url = None
#__query_lock = threading.Lock() #lock on http adapter updates

from .transport.adapter import SerializingAdapter
from .transport.network import HttpConnection
from .util import  min_args, max_args
from .util.zip_packer import Packer
from .cloud import CloudException
from cloud import _getcloudnetconnection, _getcloud

cloudLog = logging.getLogger('Cloud.files')

_file_new_query = 'file/new/'
_file_put_query = 'file/put/'
_file_list_query = 'file/list/'
_file_get_query = 'file/get/'
_file_exists_query = 'file/exists/'
_file_md5_query = 'file/md5/'
_file_delete_query = 'file/delete/'
_filemap_job_query = 'job/filemap/'
_default_chunk_size = 200

"""
This module utilizes the cloud object extensively
The functions can be viewed as instance methods of the Cloud (hence accessing of protected variables)
"""

def _post(conn, url, post_values, headers={}):
    """Use HttpConnection *conn* to issue a post request at *url* with values *post_values*"""
    
    #remove UNICODE from addresses
    url = url.decode('ascii', 'replace').encode('ascii', 'replace')
    
    if 'success_action_redirect' in headers:
        headers['success_action_redirect'] = headers['success_action_redirect'].decode('ascii', 'replace').encode('ascii', 'replace')
    if post_values and 'success_action_redirect' in post_values:
        post_values['success_action_redirect'] = post_values['success_action_redirect'].decode('ascii', 'replace').encode('ascii', 'replace')
    
    cloudLog.debug('post url %s with post_values=%s. headers=%s' % (url, post_values, headers))
    response =  conn.post(url, post_values, headers, use_gzip=False)
    
    return response

def _aws_retryable_post(conn, url, post_values, headers={}):
    """Wraps _post with ability to retry"""
    retry_attempts = conn.retry_attempts
    attempt = 0
    while attempt <= retry_attempts:    
        try:
            return _post(conn, url, post_values, headers)
        except Exception, e:
            if isinstance(e, urllib2.HTTPError) and 200 < e.code < 300: # python 2.5 bug
                return e
            
            attempt += 1            
            if attempt > retry_attempts:
                cloudLog.exception('_aws_retryable_post: Cannot connect to AWS')
                raise 
            cloudLog.warn('_aws_retryable_post: Problem connecting to AWS. Retrying. \nError is %s' % str(e))
            c = attempt -1 
            if (isinstance(e, socket.error) and getattr(e, 'errno', e.args[0]) == errno.ECONNREFUSED) or \
                (isinstance(e, urllib2.HTTPError) and e.code in [500, 503]):
                # guarantee at least a 3 second sleep on a connection refused error/500
                time.sleep(min(60, 3 + (1 << c) *random.random()))
            else:
                time.sleep(min(30, (1 << c) *random.random()))
            continue
            
        

class CloudFile(object):
    """A CloudFile represents a file stored on PiCloud as a readonly file-like stream.
    Seeking is not available."""
    
    __http_response = None

    __file_size = None
    __start_byte = None
    __end_byte = None
    __pos = None
    
    def __init__(self, http_response, file_size, start_byte=0, end_byte=None):
        self.__http_response = http_response
        
        self.__file_size = file_size
        self.__start_byte = start_byte
        self.__pos = self.__start_byte
        if end_byte:
            self.__end_byte = min(end_byte, file_size)
        else:
            self.__end_byte = file_size
                
    def __iter__(self):
        return self

    def close(self):        
        """
        Close the file, blocking further I/O operations
        """
        return self.__http_response.close()
    
    @property
    def md5(self):
        """The md5 checksum of file contents"""
        return self.__http_response.headers['etag'].strip('"')
    
    def next(self):
        data = self.__http_response.next()
        self.__pos += len(data)
        return data
    
    def read(self, size=-1):
        """
        read([size]) -> read at most size bytes, returned as a string.
       
        If the size argument is negative or omitted, read until EOF is reached.
        """
        data = self.__http_response.read(size)
        self.__pos += len(data)
        return data
        
    def readline(self, size=-1):
        """
        readline([size]) -> next line from the file, as a string.
       
        Retain newline.  A non-negative size argument limits the maximum
        number of bytes to return (an incomplete line may be returned then).
        Return an empty string at EOF.
        """
        line = self.__http_response.readline(size)
        self.__pos += len(line)
        return line
        
    def readlines(self, sizehint=0):
        """
        readlines([size]) -> list of strings, each a line from the file.
       
        Call readline() repeatedly and return a list of the lines so read.
        The optional size argument, if given, is an approximate bound on the
        total number of bytes in the lines returned.
        """
        lines = self.__http_response.readlines(sizehint)
        self.__pos += sum([len(line) for line in lines])
        return lines

    def filesize(self):
        return self.__file_size
    
    def sizeofchunk(self):
        return self.__start_byte - self.__end_byte
    
    def tell(self):
        '''Returns current file position as an integer'''
        return self.__pos

    def byte_range(self):
        return self.__byte_range

    def end(self):
        return self.__end_byte
 
def _compute_md5(f, buffersize = 8192):
    """Computes the md5 hash of file-like object
    f must have seek ability to perform this operation
    
    buffersize controls how much of file is read at once"""
    
    try:
        start_loc = f.tell()
        f.seek(start_loc)  #test seek
    except (AttributeError, IOError), e:
        raise IOError('%s is not seekable. Cannot compute MD5 hash. Exact error is %s' % (str(f), str(e)))
    
    try:
        from hashlib import md5
    except ImportError:
        from md5 import md5
    
    m = md5()    
    s = f.read(buffersize)
    while s:
        m.update(s)
        s = f.read(buffersize)
    
    f.seek(start_loc)
    hex_md5 = m.hexdigest()
    return hex_md5        

def put(source, name=None):
    """Transfers the file specified by ``file_path`` to PiCloud. The file
    can be retrieved later using the get function.    
    
    If ``name`` is specified, the file will be stored as name on PiCloud.
    Otherwise it will be stored as the basename of file_path.
    
    Example::    
    
        cloud.files.put('data/names.txt') 
    
    This will transfer the file from the local path 'data/names.txt'
    to PiCloud and store it as 'names.txt'.
    It can be later retrieved via cloud.files.get('names.txt')"""

    if not name:
        name = os.path.basename(source)
    
    # open the requested file in binary mode (relevant in windows)
    f = open(source, 'rb')
    
    putf(f, name)


def putf(f, name):
    """Similar to put.
    putf, however, accepts a file object (file, StringIO, etc.) ``f`` instead of a file_path.
    
    .. note::
        
        ``f`` is not rewound. f.read() from current position will be placed on PiCloud
    
    .. warning:: 
    
        If the file object does not correspond to an actual file on disk,
        it will be read entirely into memory before being transferred to PiCloud."""
    
    if '../..' in name:
        raise ValueError('"../.." cannot be in name')
    
    fsize = 0 # file size. may not be computable 
    
    if isinstance(f, basestring):
        fsize = len(f)                        
        from cStringIO import StringIO        
        f = StringIO(f)
    else:
        try:
            fsize = os.fstat(f.fileno()).st_size
        except (AttributeError, OSError):
            pass
    
    if fsize > 5000000000:
        raise ValueError('Cannot store files larger than 5GB on cloud.files')
    
    conn = _getcloudnetconnection()         
    
    try:
        # get a file ticket
        resp = conn.send_request(_file_new_query, {'name': name})
        ticket = resp['ticket']
        params = resp['params']
        
        url = params['action']
        
        # set file in ticket
        ticket['file'] = f
        
        # post file using information in ticket
        ticket['key'] = str(ticket['key'])
        resp =  _aws_retryable_post(conn, url, ticket)
        resp.read()
        
    finally:
        f.close()

def sync_to_cloud(source, name=None):
    """Upload file if it has changed.
    
    cloud.files.put(source,name) 
    only if contents of local file (specified by *source*)
    differ from those on PiCloud (specified by *name* or basename(*source*))
    (or if file does not exist on PiCloud)"""
    if not name:
        name = os.path.basename(source)
    
    # open the requested file in binary mode (relevant in windows)
    f = open(source, 'rb')
    local_md5 = _compute_md5(f)
    try:
        remote_md5 = get_md5(name, log_missing_file_error = False)
    except CloudException: #file not found
        remote_md5 = ''
    
    do_update = remote_md5 != local_md5
    cloudLog.debug('remote_md5=%s. local_md5=%s. uploading? %s',
                   remote_md5, local_md5, do_update
                   )
    if do_update:
        putf(f, name)
 
def list():
    """List all files stored on PiCloud."""
    
    conn = _getcloudnetconnection()

    resp = conn.send_request(_file_list_query, {})
    files = resp['files']

    return files

def _file_info(name):
    """
    get information about name
    """
    conn = _getcloudnetconnection()
    
    resp = conn.send_request(_file_exists_query, {'name':name})
    return resp

def exists(name):
    """Check if a file named ``name`` is stored on PiCloud."""
    conn = _getcloudnetconnection()
        
    resp = conn.send_request(_file_exists_query, {'name': name})
    exists = resp['exists']
    return exists

def get_md5(name, log_missing_file_error = True):
    """Return the md5 checksum of the file named ``name`` stored on PiCloud"""
    conn = _getcloudnetconnection()
    resp = conn.send_request(_file_md5_query, {'name': name},
                             log_cloud_excp = log_missing_file_error)
    md5sum = resp['md5sum']
    return md5sum
    
def delete(name):
    """Deletes the file named ``name`` from PiCloud."""
    conn = _getcloudnetconnection()

    resp = conn.send_request(_file_delete_query, {'name': name})
    deleted = resp['deleted']
    return deleted
    
def get(name, destination=None, start_byte=0, end_byte=None):
    """
    Retrieves the file named by ``name`` from PiCloud and stores it to ``destination``.
        
    Example::    
    
        cloud.files.get('names.txt','data/names.txt') 
    
    This will retrieve the 'names.txt' file on PiCloud and save it locally to 
    'data/names.txt'. 

    If destination is None, destination will be name
    
    An optional byte_range can be specified using *start_byte*, *end_byte*,
    where only the data between *start_byte* and *end_byte* is returned. 
    If end_byte exceeds the size of the file, the contents from *start_byte* to end of file returned.
    
    An end_byte of None or exceeding file size is interpreted as end of file
    """
    
    if not destination:
        destination = name
        
    cloud_file = getf(name, start_byte, end_byte)
    
    chunk_size = 64000
    f = open(destination, 'wb')
    
    while True:
        data = cloud_file.read(chunk_size)
        if not data:
            break
        f.write(data)
    
    f.close()

    
def sync_from_cloud(name, destination=None):
    """Download file if it has changed.
    
    cloud.files.get(name,destination) 
    only if contents of local file (specified by *destination* or basename(*name))
    differ from those on PiCloud (specified by *name*) 
    (or *destination* does not exist locally)"""
    
    if not destination:
        destination = name
    
    if not os.path.exists(destination):
        do_update = True
    else:
        f = open(destination, 'rb')
        local_md5 = _compute_md5(f)
        f.close()
        try:
            remote_md5 = get_md5(name, log_missing_file_error=False)
        except CloudException: #file not found
            remote_md5 = ''
    
        do_update = remote_md5 != local_md5
        cloudLog.debug('remote_md5=%s. local_md5=%s. downloading? %s',
                       remote_md5, local_md5, do_update)
    if do_update:
        get(name, destination)

def getf(name, start_byte=0, end_byte=None):
    """
    Retrieves the file named by ``name`` from PiCloud.
    Return value is a CloudFile (file-like object) that can be read() to retrieve the file's contents 

    A range can be specified through *start_byte* and *end_byte*, where only the data between those two offsets
    will be accessable in the CloudFile.  If start_byte is set, the returned CloudFile.tell() will be start_byte
    
    An end_byte of None or exceeding file size is interpretted as end of file
    """    
    
    conn = _getcloudnetconnection()

    resp = conn.send_request(_file_get_query, {'name': name})
    
    ticket = resp['ticket']
    params = resp['params']
    file_size = params['size']
    
    if not start_byte:
        start_byte=0
    if file_size and (not end_byte or end_byte > file_size):
        end_byte = file_size

    if not isinstance(start_byte, (int, long)):
        raise TypeError('start_byte must be an integer')
    
    if end_byte and not isinstance(end_byte, (int, long)):
        raise TypeError('end_byte must be an integer')

    if end_byte:
        ticket['Range'] = 'bytes=%s-%s' % tuple(  [start_byte, end_byte]  )

    resp =  _aws_retryable_post(conn, params['action'], None, ticket)
    
    cloud_file = CloudFile( resp, file_size, start_byte, end_byte )
    
    return cloud_file


def default_record_reader(delimiter):

    def def_record_reader(filesplit_obj, end_byte):

        start = filesplit_obj.tell()
        end = end_byte

        if start==0:
            skipfirstline = False       # we are reading the head of the file
        else:
            skipfirstline = True        # we are reading the middle of a chunk returned in another job
 
        to_be_searched_buffer = ''      # the string that will be searched for the delimiter
        record_buffer = ''              # Record that has been read so far 
        record_start = start            # pointer to the beginning of the record 
        crossed_end_of_chunk = False
        
        while filesplit_obj.tell() < filesplit_obj.filesize():
            
            to_be_searched_buffer = filesplit_obj.read(1024)

            while True:
                
                partition = to_be_searched_buffer.partition(delimiter)
                
                if len(partition[1])==0: # delimiter not found
                    record_buffer = record_buffer + to_be_searched_buffer
                    record_start = record_start + len(to_be_searched_buffer)
                
                    if filesplit_obj.tell()==filesplit_obj.filesize():  # are we at EOF?
                        if not skipfirstline:
                            yield record_buffer
                    
                    break
                
                else:                   # delimiter has been found
                    if skipfirstline:
                        skipfirstline = False
                    else:
                        yield record_buffer + partition[0]
                        
                    index = len(partition[0])
                    if end < (record_start + index):
                        crossed_end_of_chunk = True
                        break
                    
                    to_be_searched_buffer = partition[2]
                    record_buffer=''
                    record_start = record_start + index + 1
                        
            if crossed_end_of_chunk:
                break
         
        filesplit_obj.close()
        
    return def_record_reader


def _validate_arguments(func_arg, param_name):
    """Validate that certain map parameters are callables that take 1 argument"""
    if not callable(func_arg):
        raise TypeError( '%s argument (%s) must be callable'  % (param_name, str(func_arg)) )
    
    try:
        max_arg = max_args(func_arg)
        min_arg = min_args(func_arg)
    except TypeError:
        pass #type can't be sanity checked.. let it through
    else:
        if max_arg < 1 or min_arg > 1:
            raise TypeError( '%s argument (%s) must accept one parameter'  % (param_name, str(func_arg)) )            

def _validate_rr_arguments(func_arg, param_name):
    """Validate that certain map parameters are callables that take 1 argument"""
    if not callable(func_arg):
        raise TypeError( '%s argument (%s) must be callable'  % (param_name, str(func_arg)) )
    
    try:
        max_arg = max_args(func_arg)
        min_arg = min_args(func_arg)
    except TypeError:
        pass #type can't be sanity checked.. let it through
    else:
        if max_arg < 2 or min_arg > 2:
            raise TypeError( '%s argument (%s) must accept two parameters'  % (param_name, str(func_arg)) )            

def _mapper_combiner_wrapper(mapper, name, file_size, record_reader, combiner):
    
    def inner(server_mapper):
        return server_mapper(mapper, name, file_size, record_reader, combiner)
    
    return inner

def _reducer_wrapper(reducer):
    
    def inner(server_reducer):
        return server_reducer(reducer)
    
    return inner

def map(name, mapper, chunk_size=None, record_reader=None, combiner=None, reducer=None, **kwargs):
    """
    With map, you can process a file stored in cloud.files in parallel. The
    parallelism is achieved by dividing the file specified by *name* into
    chunks of size *chunk_size* (bytes). Each chunk is assigned a sub job. The
    sub job in turn processes just that chunk, allowing for the entire file to
    be processed by as many cores in parallel as there are chunks. We will call
    this type of sub job a "mapper sub job".
    
    If chunk_size is None, it will be automatically set to 1/10th of the size
    of the file.
    
    Map will return a single job IDentifier (jid). The sub jobs that comprise it
    do not have identifiers and, therefore, cannot be accessed directly.
    cloud.info(jid), however, will show you information for relevant sub jobs.
    
    By default, each chunk is split into records (of 0 or more characters) using
    newlines as delimiters. If *record_reader* is specified as a string, each
    chunk is split into records using that as the delimiter.
    
    In the event a record spans across two chunks, it is guaranteed that a mapper
    will only be called once on the full record. In other words, we've made sure
    it works correctly.
    
    *mapper* is a function that takes a single argument, a record, and should
    return an iterable of values (a generator). In the simplest case, it can
    return a generator that yields only one value.
    
    Example::
    
        def mapper(record):
            yield record
    
    When no *combiner* or *reducer* is specified, the return value of the
    cloud.files.map job will be roughly equivalent to::
            
            map(mapper, record_reader(file_contents))
    
    A *reducer* is a function that takes in an iterable of values and returns an 
    iterable of values.  The iterable parameter iterates through all the values 
    returned by all the mapper(record) calls. When the reducer is specified,
    *reducer* will result in the creation of one additional sub job. The reducer
    sub job grabs the results of each mapper sub job (iterators), combines them
    into a single iterator, and then passes that iterator into your *reducer*
    function. The return value of the cloud.files.map job will be the iterator
    returned by the *reducer*.
    
    A *combiner*, like a *reducer*, takes in an iterable of values and returns an
    iterable of values. The difference is that the *combiner* is run in each
    mapper sub job, and each one only takes in values that were produced from the
    associated chunk. If a *reducer* is also specified, then the reducer sub job
    grabs the results of each *combiner* run in each mapper sub job.
    
    Example for counting the number of words in a document::
    
        def wordcount_mapper(record):
            yield len(record.split(' '))
            
        def wordcount_reducer(wordcounts):
            yield sum(wordcounts)
            
        jid = cloud.files.map('example_document', wordcount_mapper, reducer=wordcount_reducer)
        
    Result::
        cloud.result(jid)
            >> [# of words]
    
    For advanced users, *record_reader* can also be specified as a function that
    takes in a file-like object (has methods read(), tell(), and seek()), and
    the end_byte for the current chunk. The *record_reader* should return an
    iterable of records.  See default_record_reader for an example.
    
    Additional information exists on our blog and online documentation.
        
        Reserved special *kwargs* (see docs for details):
        
        * _cores:
            Set number of cores your job will utilize. See http://blog.picloud.com/2012/08/31/introducing-multicore-support/
            In addition to having access to more cores, you will have _cores*RAM[_type] where _type is the _type you select
            Possible values depend on what _type you choose:
            
            'c1': 1
            'c2': 1, 2, 4, 8
            'f2': 1, 2, 4, 8, 16
            'm1': 1, 2
            's1': 1        
        * _depends_on:
            An iterable of jids that represents all jobs that must complete successfully 
            before any jobs created by this map function may be run.
        * _depends_on_errors:
            A string specifying how an error with a jid listed in _depends_on should be handled.
            'abort': Set this job to 'stalled'  (Default)
            'ignore': Treat an error as satisfying the dependency            
        * _env:
            A string specifying a custom environment you wish to run your jobs within.
            See environments overview at 
            http://blog.picloud.com/2011/09/26/introducing-environments-run-anything-on-picloud/
        * _fast_serialization:
            This keyword can be used to speed up serialization, at the cost of some functionality.
            This affects the serialization of both the map arguments and return values
            The map function will always be serialized by the enhanced serializer, with debugging features.
            Possible values keyword are:
                        
            0. default -- use cloud module's enhanced serialization and debugging info            
            1. no debug -- Disable all debugging features for arguments            
            2. use cPickle -- Use python's fast serializer, possibly causing PicklingErrors                
        * _kill_process:
                Terminate the Python interpreter *func* runs in after *func* completes, preventing
                the interpreter from being used by subsequent jobs.  See Technical Overview for more info.                            
        * _label: 
            A user-defined string label that is attached to the created jobs. 
            Labels can be used to filter when viewing jobs interactively (i.e.
            on the PiCloud website).        
        * _max_runtime:
            Specify the maximum amount of time (in integer minutes) a job can run. If job runs beyond 
            this time, it will be killed.                     
        * _priority: 
                A positive integer denoting the job's priority. PiCloud tries to run jobs 
                with lower priority numbers before jobs with higher priority numbers.            
        * _profile:
                Set this to True to enable profiling of your code. Profiling information is 
                valuable for debugging, but may slow down your job.
        * _restartable:
                In the very rare event of hardware failure, this flag indicates that the job
                can be restarted if the failure happened in the middle of the job.
                By default, this is true. This should be unset if the job has external state
                (e.g. it modifies a database entry)
        * _type:
                Select the type of compute resources to use.  PiCloud supports four types,
                specified as strings:
                
                'c1'
                    1 compute unit, 300 MB ram, low I/O (default)                    
                'c2'
                    2.5 compute units, 800 MB ram, medium I/O                    
                'm1'                    
                    3.25 compute units, 8 GB ram, high I/O
                's1'
                    variable compute units (2 cu max), 300 MB ram, low I/O, 1 IP per core                    
                                    
                See http://www.picloud.com/pricing/ for pricing information
    """
    
    cloud_obj = _getcloud()
    params = cloud_obj._getJobParameters(mapper, kwargs)    # takes care of kwargs
    
    file_details = _file_info(name)
    if not file_details['exists']:
        raise ValueError('file does not exist on the cloud, or is not yet ready to be accessed')
    file_size = int( file_details['size'] )
    params['file_name'] = name
    
    
    # chunk_size
    if chunk_size:
        if chunk_size==0:
            raise Exception('the chunk_size should be a non zero integer value')
        if not isinstance(chunk_size, (int, long)):
            raise Exception('the chunk_size should be a non zero integer value')        
        params['chunk_size'] = chunk_size
            
    
    # mapper
    _validate_arguments(mapper, 'mapper')
    
    # record_reader
    if not record_reader:
        record_reader = default_record_reader('\n')
    else:
        if isinstance(record_reader, basestring):
            record_reader = default_record_reader(record_reader)
        else:
            _validate_rr_arguments(record_reader, 'record_reader')
    
    # combiner
    if not combiner:
        def combiner(it):
            for x in it:
                yield x
    else:
        _validate_arguments(combiner, 'combiner')


    func_to_be_sent = _mapper_combiner_wrapper(mapper, name, file_size, record_reader, combiner)
    
    sfunc, sarg, logprefix, logcnt = cloud_obj.adapter.cloud_serialize( func_to_be_sent, 
                                                                    params['fast_serialization'], 
                                                                    [], 
                                                                    logprefix='mapreduce.' )
    
    data = Packer()
    data.add(sfunc)
    params['data'] = data.finish()
    
    # validate reducer & serialize reducer
    if reducer:
        _validate_arguments(reducer, 'reducer')
        reducer = _reducer_wrapper(reducer)
        s_reducer, red_sarg, red_logprefix, red_logcnt = cloud_obj.adapter.cloud_serialize( reducer, params['fast_serialization'], [], logprefix='mapreduce.reducer.' )
        data_red = Packer()
        data_red.add(s_reducer)
        params['data_red'] = data_red.finish()
        
    conn = _getcloudnetconnection()
    conn._update_params(params)
    cloud_obj.adapter.dep_snapshot()
    
    resp = conn.send_request(_filemap_job_query, params)
    
    return resp['jids']
