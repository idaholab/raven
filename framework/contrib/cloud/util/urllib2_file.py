#!/usr/bin/env python
####
# Heavily modified version of urllib2_file. Original docs follow:
# Version 1.0 (PiCloud):
#  - Support gzip compression each way, use python2.5 http interface, etc.
#  - Add support for persistent HTTP Connections
# Version: 0.2.0
#  - UTF-8 filenames are now allowed (Eli Golovinsky)<br/>
#  - File object is no more mandatory, Object only needs to have seek() read() attributes (Eli Golovinsky)<br/>
#
# Version: 0.1.0
#  - upload is now done with chunks (Adam Ambrose)
#
# TODO: This code is incredibly sloppy. Rewrite one day (Boto connection is better starting point) 

# Version: older
# THANKS TO:
# bug fix: kosh @T aesaeion.com
# HTTPS support : Ryan Grow <ryangrow @T yahoo.com>
# Copyright (C) 2004,2005,2006 Fabien SEISEN
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
# 
# you can contact me at: <fabien@seisen.org>
# http://fabien.seisen.org/python/
#
# Also modified by Adam Ambrose (aambrose @T pacbell.net) to write data in
# chunks (hardcoded to CHUNK_SIZE for now), so the entire contents of the file
# don't need to be kept in memory.
#

"""
enable to upload files using multipart/form-data

idea from:
upload files in python:
 http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/146306

timeoutsocket.py: overriding Python socket API:
 http://www.timo-tasi.org/python/timeoutsocket.py
 http://mail.python.org/pipermail/python-announce-list/2001-December/001095.html

import urllib2_files
import urllib2
u = urllib2.urlopen('http://site.com/path' [, data])

data can be a mapping object or a sequence of two-elements tuples
(like in original urllib2.urlopen())
varname still need to be a string and
value can be string of a file object
eg:
  ((varname, value),
   (varname2, value),
  )
  or
  { name:  value,
    name2: value2
  }

"""

import os
import socket
import sys
import stat
import thread
import mimetypes
import errno
#BUG in python 2.6 mimetypes: We must call init() here to ensure thread safety
if not mimetypes.inited:
    mimetypes.init()

import random
import httplib
import urllib
import urllib2

try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

import logging
cloudLog = logging.getLogger('Cloud.HTTPConnection')


CHUNK_SIZE = 65536


#PiCloud injection for gzip control
from .. import cloudconfig as cc

use_gzip = cc.transport_configurable('use_gzip',
                                     default=True,hidden=False,
                                     comment='Request gziped HTTP responses')
client_gzip = False #gzip http requests? (genreally don't use this)

http_close_connection = cc.transport_configurable('http_close',
                                     default=False,hidden=False,
                                     comment='Should every HTTP connection be closed after receiving a response?')
c = """Specify an http/https proxy server that should be used
e.g. proxy.example.com:3128 for anonymous  or username:password@proxy.example.com:3128 for authenticated
"""
proxy_server = cc.transport_configurable('proxy_server',default='',comment = c) 

#new:
from .gzip_stream import GzipFile
#from gzip import GzipFile


def choose_boundary():
    """Choose boundary with more randomness
    Pray that it doesn't conflict :)"""
    import email.generator as generator
    _fmt = generator._fmt
    boundary = ('=' * 8)
    for i in range(8):          
        token = random.randrange(sys.maxint)
        boundary += _fmt % token
    boundary += '=='
    return boundary

def get_content_type(filename):
    return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

def send_data(v_vars, v_files, boundary, gzip = True, sock = None):
    """Either returns data or sends in depending on arguments.
    
    Return value is a tuple of string data, content-length
    
    Content-length always returns; it represents the content-length of the http request
    
    If sock is None:
        Typically string data is returned; this can be written straight to the socket
        
        However, as an optimization, if any v_files correspond to a file on disk and gzip is false,
            The data returned will be none. This must then call send_data again with sock nonnull
    
    If sock is a connection, content-length will be returned AND data will be streamed over socket (but not returned) 
        direct socket writing will not work if gzip is True as content-length of http request will be wrong
    """
    
    all_in_memory = True  #if we can return this
    cl = 0 #holds content_length
    
    if gzip: 
        if sock:
            raise TypeError('gzip and sock cannot both be true')  
        gzip_file = StringIO()
        buffer = GzipFile(filename=None,fileobj=gzip_file,mode='w')
    else:
        buffer = StringIO()

    
    for (k, v) in v_vars:
        buffer.write('--%s\r\n' % boundary)
        buffer.write('Content-Disposition: form-data; name="%s"\r\n\r\n%s\r\n' % (k,v))
    
    for (k, v) in v_files:
        fd = v
        cur_pos = None
        #print 'transmit %s %s' % (k,v)
        contents = None
        try:  #check if String IO
            contents = fd.getvalue()
            file_size = len(contents)
            name = k
        except AttributeError, e:
            #This might be a file on disk
            try:
                
                assert not gzip #if gzipping is true, we must fall back to reading in file
                cur_pos = fd.tell()
                fd.seek(0)  # test rewind ability
                fd.seek(0,2)  
                file_size = fd.tell() - cur_pos
                #print 'f %s %s' % (file_size, cur_pos)
                fd.seek(cur_pos)  # back to original position
                
            except (AssertionError, AttributeError, OSError), e: 
                # must abort when file not seekable, as re-try (in network.send_request) impossible
                raise ValueError('File postdata %s is not seekable. Cannot transfer' % k)
            
            try:
                name = fd.name.split('/')[-1]
                if isinstance(name, unicode):
                    name = name.encode('UTF-8')
            except AttributeError: #no name? revert to k 
                name = k

        buffer.write('--%s\r\n' % boundary)
        buffer.write('Content-Disposition: form-data; name="%s"; filename="%s"\r\n' \
                  % (k, name))
        buffer.write('Content-Type: %s\r\n' % get_content_type(name))
        buffer.write('Content-Length: %s\r\n' % file_size)        
        
        buffer.write('\r\n')

        if not contents:
            all_in_memory = False
        
        #print '[%s] transmit %s %s - in mem? %s contents? %s' % (sock, k,v, all_in_memory, bool(contents))
        if sock or not all_in_memory:
            assert not gzip #sanity check
            
            #reset buffer and update content-length if not returning result                
            buf_str = buffer.getvalue()            
            buffer.close()   
            cl += len(buf_str)
            buffer = StringIO() #regen buffer
            cl += file_size  
            
            if sock:             
                sock.sendall(buf_str)
                
                if contents:
                    sock.sendall(contents)
                else:
                    data=fd.read(CHUNK_SIZE)
                    try:
                        while data:
                            sock.sendall(data)
                            data=fd.read(CHUNK_SIZE)
                    finally:
                        if cur_pos != None: 
                            # rewind so retry will work
                            fd.seek(cur_pos)
            
        
        elif all_in_memory: #keep writing into RAM
            buffer.write(contents)
            
        buffer.write('\r\n')
            
    buffer.write('--%s--\r\n\r\n' % boundary)   
    
    if sock or not all_in_memory:
        buf_str = buffer.getvalue()
        if sock:
            sock.sendall(buf_str)        
        cl += len(buf_str)
        return None, cl        
    
    #in memory
    #When Using compressed content_len = compressed length (gzip
    # we need mod_wsgi middleware on the django server
    if gzip:
        buffer.close()
        buf_str = gzip_file.getvalue()
    else:        
        buf_str = buffer.getvalue()    
    
    return buf_str, len(buf_str)

def makeBodyFunction(v_vars, v_files, boundary):
    def theBody(httpcon):
        try:
            send_data(v_vars, v_files, boundary, gzip = False, sock = httpcon.sock)
        except socket.error, v:
            if v[0] == 32:      # Broken pipe
                httpcon.close()
            raise
    return theBody


class funcBodyHTTPConnection(httplib.HTTPConnection):
    """
    httplib connection that supports calling a function to handle body sending
    body takes a single argument -- this connection. 
    Note that automatic content-length calculation is not done if body is a function
    
    Also supports persistant
    """
    
    def _send_request(self, method, url, body, headers):
    
        if callable(body):
            httplib.HTTPConnection._send_request(self, method, url, None, headers)
            body(self)
        else:
            httplib.HTTPConnection._send_request(self, method, url, body, headers)
            
    @staticmethod
    def getresponse_static(self):
                
        """Get the response from the server.
        This is a static method used by getresponse in http and https"""
        
        #modified to read any extra data from pipe

        # if a prior response has been completed, then forget about it.
        if self._HTTPConnection__response:
            closed = self._HTTPConnection__response.isclosed()

            
            if not closed:
                self._HTTPConnection__response.read()
            
            self._HTTPConnection__response = None

        #
        # if a prior response exists, then it must be completed (otherwise, we
        # cannot read this response's header to determine the connection-close
        # behavior)
        #
        # note: if a prior response existed, but was connection-close, then the
        # socket and response were made independent of this HTTPConnection
        # object since a new request requires that we open a whole new
        # connection
        #
        # this means the prior response had one of two states:
        #   1) will_close: this connection was reset and the prior socket and
        #                  response operate independently
        #   2) persistent: the response was retained and we await its
        #                  isclosed() status to become true.
        #
        if self._HTTPConnection__state != httplib._CS_REQ_SENT or self._HTTPConnection__response:
            #print 'not ready %s %s' % (self._HTTPConnection__state, self._HTTPConnection__response)
            raise httplib.ResponseNotReady('State is %s' % self._HTTPConnection__state) 

        if self.debuglevel > 0:
            response = self.response_class(self.sock, self.debuglevel,
                                           strict=self.strict,
                                           method=self._method)
        else:
            response = self.response_class(self.sock, strict=self.strict,
                                           method=self._method)

        response.begin()
        assert response.will_close != httplib._UNKNOWN
        self._HTTPConnection__state = httplib._CS_IDLE

        if response.will_close:            
            # this effectively passes the connection to the response
            self.close()
            
            #hax:
            #self._HTTPConnection__response = response
            
        else:
            # remember this, so we can tell when it is complete
            self._HTTPConnection__response = response

        return response    
    
    def getresponse(self):
        return self.getresponse_static(self)        
        


# modified version from urllib2
class newHTTPAbstractHandler(urllib2.AbstractHTTPHandler):

    def __init__(self, debuglevel=0):
        urllib2.AbstractHTTPHandler.__init__(self, debuglevel)
        self.connections = {}
    
    def do_open(self, http_class, req):
        """Adds ability to reuse connections
        """
        host = req.get_host()
        if not host:
            raise urllib2.URLError('no host given')
        
        effective_host = getattr(req,'_tunnel_host', None) #check proxy first
        if not effective_host:
            effective_host = req.get_host()            
        #print 'effhost is %s. req host is %s' % (effective_host, host)
        
        conn_key = effective_host, http_class, thread.get_ident()  #good enough for picloud
        #print 'conn_key is ', conn_key
        h = self.connections.get(conn_key)
            
        if h:
            #print 'CONN: found reuse conn!', h.sock            
            reusing = True
            h._set_hostport(host,None)  #proxy tunneling renames this
        if not h:  #no cached conn - reconnect
            #print 'CONN: using a new connection'
            if hasattr(req,'timeout'):
                h = http_class(host, timeout=req.timeout) # will parse host:port
            else:
                h = http_class(host)  #py2.5
            h.set_debuglevel(self._debuglevel)
            self.connections[conn_key] = h
            reusing = False
        #To enable debugging
        #h.set_debuglevel(1)
        

        headers = dict(req.headers)
        headers.update(req.unredirected_hdrs)
        # We want to make an HTTP/1.1 request, but the addinfourl
        # class isn't prepared to deal with a persistent connection.
        # It will try to read all remaining data from the socket,
        # which will block while the server waits for the next request.
        # So make sure the connection gets closed after the (only)
        # request.
        
        close_connection = http_close_connection or headers.get("Connection") == "close" 
        
        if close_connection:
            headers["Connection"] = "close"
        
        headers = dict(
            (name.title(), val) for name, val in headers.items())

        if getattr(req, '_tunnel_host', None): #py2.5 lacks this
            tunnel_headers = {}
            proxy_auth_hdr = "Proxy-Authorization"
            if proxy_auth_hdr in headers:
                tunnel_headers[proxy_auth_hdr] = headers[proxy_auth_hdr]
                # Proxy-Authorization should not be sent to origin
                # server.
                del headers[proxy_auth_hdr]
            if hasattr(h, '_set_tunnel'):  #python2.6 and below
                h._set_tunnel(req._tunnel_host, headers=tunnel_headers)
            else:  #python2.7+ removed protection
                h.set_tunnel(req._tunnel_host, headers=tunnel_headers)

        for i in range(2): 
            #persistent connection handling via httplib2 
            try:
                #hack -- check if old response closed        
                #print 'start req'        
                h.request(req.get_method(), req.get_selector(), req.data, headers)
                #print 'end req'
            except socket.gaierror, err:
                h.close()
                raise urllib2.URLError(err)
            except (socket.error, httplib.HTTPException), e:
                #may still have response
                if isinstance(e, socket.error) and getattr(e, 'errno', e.args[0]) == errno.ECONNREFUSED:
                    h.close()
                    raise
                cloudLog.info('Error raised by request. reinitiating connection. Exception was:', exc_info=1)
                h.close()
                if i == 0:
                    continue
                else:
                    raise
            except Exception: #cleanup on any request error
                h.close()
                raise 
                
            try:
                #print 'sock is', h.sock
                r = h.getresponse()
                #print 'post-end is', h.sock, r.will_close
            except Exception, err:  #known exceptions are (socket.error, httplib.HTTPException, AttributeError)
                #print 'socket closed!!', str(err)
                #socket closed -- retry?
                if i == 0 and reusing:
                #if False:
                    cloudLog.info('Reconnecting socket due to:', exc_info=1) 
                    h.close()
                    h.connect()
                    continue
                    #raise  urllib2.URLError(err)
                else:
                    raise  urllib2.URLError(err)
                    #print err
                    #raise
            break


        # Pick apart the HTTPResponse object to get the addinfourl
        # object initialized properly.

        # Wrap the HTTPResponse object in socket's file object adapter
        # for Windows.  That adapter calls recv(), so delegate recv()
        # to read().  This weird wrapping allows the returned object to
        # have readline() and readlines() methods.

        # XXX It might be better to extract the read buffering code
        # out of socket._fileobject() and into a base class.

        r.recv = r.read
        try:
            fp = socket._fileobject(r, close=close_connection)  #set close to false to leave conn open
        except TypeError: #python pre-2.5.1 lacks close kwarg
            fp = socket._fileobject(r)

        resp = urllib2.addinfourl(fp, r.msg, req.get_full_url())
        resp.code = r.status
        resp.msg = r.reason
        
        return resp


    def do_request_(self, request):
        """Modified to support multipart data"""
        
        host = request.get_host()
        if not host:
            raise urllib2.URLError('no host given')

        if not request.has_data():
            request = urllib2.AbstractHTTPHandler.do_request_(self, request)
        else:
            data = request.get_data()
            #print 'data is %s' % data
            if callable(data):
                raise TypeError('data is a callable. This is not valid!')
            v_files=[]
            v_vars=[]
            # mapping object (dict)
            if type(data) == str:
                request = urllib2.AbstractHTTPHandler.do_request_(self, request)
            else:                         
                if hasattr(data, 'items'):
                    data = list(data.items())
                else:
                    if len(data) and not isinstance(data[0], tuple):
                        raise TypeError("not a valid non-string sequence or mapping object.")
                        
                    
                for (k, v) in data:
                    if isinstance(k, unicode):
                        k = k.encode('utf-8')
                        k = urllib.quote(k)
                    
                    if hasattr(v, 'read'):
                        v_files.append((k, v))
                    else:
                        if isinstance(v, unicode):
                            v = v.encode('utf8')
                        v_vars.append( (k, v) )
                        
                if len(v_files) == 0:
                    if v_vars:
                        data = urllib.urlencode(v_vars)
                        #print 'transmit %s' % data
                    else:
                        data = ""                      
                    if not request.has_header('Content-type'):
                        request.add_unredirected_header('Content-Type',
                                    'application/x-www-form-urlencoded')
                        request.add_unredirected_header('Content-length', '%d' % len(data))
                
                elif not request.has_header('Content-type') and len(v_files) > 0:
                        boundary = choose_boundary()
                        gzip_request = client_gzip and getattr(request,'use_gzip',False)
                        
                        data, content_len = send_data(v_vars, v_files, boundary, gzip_request)
                        request.add_unredirected_header('Content-Type',
                                    'multipart/form-data; boundary=%s' % boundary)
                        request.add_unredirected_header('Content-length', str(content_len))
                        
                        #gzip outbound requests:
                        if (gzip_request):
                            request.add_unredirected_header('Content-Encoding', 'gzip')
                
                #request's data is none if we need to do further evaluation
                if data is not None:
                    request.data = data
                else: #use callback
                    request.data = makeBodyFunction(v_vars, v_files, boundary)
                            
                #final steps (from urllib2.AbstractHTTPHandler)
                sel_host = host
                if hasattr(request,'has_proxy') and request.has_proxy():
                    scheme, sel = urllib.splittype(request.get_selector())
                    sel_host, sel_path = urllib.splithost(sel)
        
                if not request.has_header('Host'):
                    request.add_unredirected_header('Host', sel_host)
                for name, value in self.parent.addheaders:
                    name = name.capitalize()
                    if not request.has_header(name):
                        request.add_unredirected_header(name, value)              

        if (use_gzip):
            request.add_unredirected_header('Accept-Encoding', 'gzip')
            #pass

        return request
        
    def do_response_(self, request, response):         
        #print 'type', fp.__class__
        if response.headers and response.headers.getheader('Content-Encoding','') == 'gzip' and\
        getattr(request,'use_gzip',False):
                                        
            fp = GzipFile(filename=None,fileobj=StringIO(response.read()),mode='r')
            #fp = GzipFile(filename=None,fileobj=response,mode='r')
            #response needs to be StringIO(response.read()) without gzip streaming

            old_response = response
            response = urllib.addinfourl(fp, old_response.headers, old_response.url)
            if hasattr(old_response,'code'):
                response.code = old_response.code
            response.msg = old_response.msg
        return response

class newHTTPHandler(newHTTPAbstractHandler):
    http_request = newHTTPAbstractHandler.do_request_
    http_response = newHTTPAbstractHandler.do_response_
    
    def http_open(self, req):
        return self.do_open(funcBodyHTTPConnection, req)    

if hasattr(httplib, 'HTTPS'):
    class funcBodyHTTPSConnection(httplib.HTTPSConnection):
        """
        See docs for funcBodyHTTPConnection
        """
        
        def _send_request(self, method, url, body, headers):
            #print 'working req is %s\nbody=%s' % (headers, body)
            if callable(body):
                #print 'callable body'
                httplib.HTTPSConnection._send_request(self, method, url, None, headers)
                body(self)
            else:
                httplib.HTTPSConnection._send_request(self, method, url, body, headers)
                
        def getresponse(self):
            return funcBodyHTTPConnection.getresponse_static(self) #use this version
        
        def _send_output(self, message_body=None):
            """Fix Python 2.7 bug"""
            self._buffer.extend(("", ""))
            msg = "\r\n".join(self._buffer)
            del self._buffer[:]
            # If msg and message_body are sent in a single send() call,
            # it will avoid performance problems caused by the interaction
            # between delayed ack and the Nagle algorithim.
            
            if isinstance(message_body, str):
                try:
                    msg += message_body
                except UnicodeDecodeError: #non-ascii
                    pass
                else:
                    message_body = None
            self.send(msg)
            if message_body is not None:
                #message_body was not a string (i.e. it is a file) 
                self.send(message_body)


    
    class newHTTPSHandler(newHTTPAbstractHandler):
        https_request = newHTTPAbstractHandler.do_request_
        https_response = newHTTPAbstractHandler.do_response_
        
        def https_open(self, req):
            return self.do_open(funcBodyHTTPSConnection, req)    

#Input handling
_opener = None

#modified urlopen - supports python 2.5 and 2.6
def urlopen(url, data=None, timeout = None):
    global _opener
    if _opener is None:
        _opener = build_opener()
    if not timeout:
        if not hasattr(socket,'_GLOBAL_DEFAULT_TIMEOUT'):
            #python 2.5
            return _opener.open(url, data)
        else:
            timeout=socket._GLOBAL_DEFAULT_TIMEOUT
    if sys.version_info < (2,6):
        return _opener.open(url, data) # version 2.5 compat 
    else:
        return _opener.open(url, data, timeout)
        

#modified build_opener:
def build_opener(*handlers):
    """Create an opener object from a list of handlers.

    The opener will use several default handlers, including support
    for HTTP and FTP.

    If any of the handlers passed as arguments are subclasses of the
    default handlers, the default handlers will not be used.
    """
    import types
    def isclass(obj):
        return isinstance(obj, types.ClassType) or hasattr(obj, "__bases__")



    opener = urllib2.OpenerDirector()
    default_classes = [urllib2.ProxyHandler, urllib2.UnknownHandler, newHTTPHandler,
                       urllib2.HTTPDefaultErrorHandler, urllib2.HTTPRedirectHandler,
                       urllib2.FTPHandler, urllib2.FileHandler, urllib2.HTTPErrorProcessor]
    if hasattr(httplib, 'HTTPS'):
        default_classes.append(newHTTPSHandler)
    skip = set()

    if proxy_server:
        handlers = list(handlers)
        handlers.append(urllib2.ProxyHandler({'http': proxy_server,
                                                 'https': proxy_server}))    
    
    for klass in default_classes:
        for check in handlers:
            if isclass(check):
                if issubclass(check, klass):
                    skip.add(klass)
            elif isinstance(check, klass):
                skip.add(klass)
    for klass in skip:
        default_classes.remove(klass)

    for klass in default_classes:
        opener.add_handler(klass())
        
    for h in handlers:
        if isclass(h):
            h = h()
        opener.add_handler(h)
    return opener
    