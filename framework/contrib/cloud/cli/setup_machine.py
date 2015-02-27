"""
Functions for setting up the local machine's API Keys.
This module is only intended for internal use
"""
"""
Copyright (c) 2012 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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

import os
import sys
import getpass
import time

import cloud
from cloud.util import credentials, fix_sudo_path
import webbrowser
import random
import BaseHTTPServer

import logging
cloudLog = logging.getLogger('Cloud.setup_machine')

WEB_AUTH_PATH = 'accounts/request_setup_token/'
WEB_TIMEOUT = 600
CALLBACK_PATH = '/auth_callback/'
SUCCESS_REDIRECT = 'accounts/received_setup_token/'

# base url to connect to PiCloud
web_base_url = None

def _is_browser_graphical(browser_obj):
    """Return if a given webbrowser.Browser object refers to a graphical browser"""
    if isinstance(browser_obj, webbrowser.UnixBrowser):
        return browser_obj.background
    if isinstance(browser_obj, webbrowser.BackgroundBrowser):
        return True
    always_graphical = ['Konqueror', 'WindowsDefault', 'MacOSX', 'MacOSXOSAScript']
    return browser_obj.__class__.__name__ in always_graphical
    

def graphical_web_open(url, new=1, autoraise=True):
    """Similar to webbrowser.open, but only displays graphical webbrowsers.
    Must use protected interfaces due to lack of an API to access such information
    
    Returns True if web was opened; False otherwise"""
    
    for name in webbrowser._tryorder:
        browser = webbrowser.get(name)
        
        if _is_browser_graphical(browser) and browser.open(url, new, autoraise):
            return True
    return False    

def system_browser_may_be_graphical():
    """Return true if there may be a graphical webbrowser on this machine"""
    if sys.platform == 'darwin' or sys.platform[:3] == 'win':
        return True
    return "DISPLAY" in os.environ

class PiCloudHTTPHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    """HTTP Server handler to retrieve email and token from PiCloud webserver
    
    Webserver issues a 302 redirect to localhost
    Credentials are inside get parameters
    """ 
    
    def log_message(self, format, *args):
        """Don't write to stderr"""
        cloudLog.debug(format, *args)
    
    def do_GET(self):
        global web_base_url
        
        # Parse out email and temporary authorization key
        valid = 0
        path_tuple = self.path.split('?')
        if len(path_tuple) == 2 and path_tuple[0] == CALLBACK_PATH:
            get_vars = path_tuple[1]
            all_gets = get_vars.split('&')
            for getbind in all_gets:
                get_tuple = getbind.split('=')
                if len(get_tuple) == 2:
                    if get_tuple[0] == 'token':
                        self.server.auth_token = get_tuple[1]
                        cloudLog.debug('Received token from server')
                        valid += 1
                if get_tuple[0] == 'email':
                    self.server.email = get_tuple[1]
                    cloudLog.debug('Received email from server: %s' % self.server.email)
                    valid += 1                
                    
        if  valid == 2: # redirect to web page telling user to close
            self.send_response(302)
            self.send_header('Location',web_base_url + SUCCESS_REDIRECT)
            self.end_headers()
        else:
            self.send_error(404, 'Invalid path')

def start_http_server():
    """Returned spawn server object"""    
    for port in xrange(30000+random.randint(0,500), 31000):
        # create listener socket, bind it to the port, and start listening
        try:
            server_address = ('localhost', port)
            httpd = BaseHTTPServer.HTTPServer(server_address, PiCloudHTTPHandler)
            # configure variables response handler writes to
            httpd.email = None
            httpd.auth_token = None            
            
            httpd.timeout = WEB_TIMEOUT
            cloudLog.debug('Listening server started on port %s' % httpd.server_port)
            return httpd
        except Exception, e:
            if e[0] == 98:   # error code corresponds to socket in use
                cloudLog.debug('socket %s already in use', port)
                continue
            cloudLog.error('failed to listen on port %s', port)
            raise    

def web_acquire_token():
    """Acquire the secure token via the browser"""
    global web_base_url
    try:
        server = start_http_server()
    except Exception, e:
        print 'Could not start local webserver due to %s!' % e,
        return
    
    api_url = cloud._getcloudnetconnection().url
    web_base_url = api_url.replace('//api.', '//') # map to web url from api    
    #web_base_url = 'http://localhost:8000/' # temp! 
    
    full_url = web_base_url + WEB_AUTH_PATH + '?redirect_port=%s' % server.server_port 
    if not graphical_web_open(full_url):
        print 'Could not launch webbrowser!',
        return
    
    # TOOD: Catch timeout
    start_time = time.time()
    while not server.auth_token:
        if time.time() > start_time + WEB_TIMEOUT:
            print 'Did not receive credentials within %s seconds.' % WEB_TIMEOUT
            break
                    
        server.handle_request()
    server.server_close()
    return server.email, server.auth_token
    

def setup_machine(email=None, password=None, api_key=None):
    """Prompts user for login information and then sets up api key on the
    local machine
    
    If api_key is a False value, interpretation is:
        None: create api key
        False: Prompt for key selection
    
    """
    
    # Disable simulator -- we need to initiate net connections
    cloud.config.use_simulator = False
    cloud.config.commit()
    
    auth_token = None # authentication key derived from webserver injection
    
    # connect to picloud
    cloud._getcloud().open()
    
    interactive_mode = not (email and password)
    
    if system_browser_may_be_graphical() and interactive_mode:
        print 'To authenticate your computer, a web browser will be launched.  Please login if prompted and follow instructions on screen.\n'
        raw_input('Press ENTER to continue')
        
        
        auth_info = web_acquire_token()
        if not auth_info:
            print 'Reverting to email/password authentication.\n'
        else:
            email, password = auth_info
            print '\n'
    
    if interactive_mode:
        print 'Please enter your PiCloud account login information.\nIf you do not have an account, please create one at http://www.picloud.com\n' + \
        'Note that a password is required. If you have not set one, set one at https://www.picloud.com/accounts/settings/\n'
    
    try:
        if email:            
            print 'Setup will proceed using this E-mail: %s' % email
        else:
            email = raw_input('E-mail: ')
            
        if not password:
            password = getpass.getpass('Password: ')
        
        if not api_key:   
            
            if interactive_mode:                
                keys = cloud.account.list_keys(email, password, active_only=True)
                
                print """\nPiCloud uses API Keys, rather than your login information, to authenticate
    your machine. In the event your machine is compromised, you can deactivate
    your API Key to disable access. In this next step, you can choose to use
    an existing API Key for this machine, or create a new one. We recommend that
    each machine have its own unique key."""
                
                print '\nYour API Key(s)'
                for key in keys:
                    print key
                
                api_key = raw_input('\nPlease select an API Key or just press enter to create a new one automatically: ')
                if api_key:
                    key = cloud.account.get_key(email, password, api_key)
                else:
                    key = cloud.account.create_key(email, password)
                    print 'API Key: %s' % key['api_key']
            else:
                api_key = cloud.config.api_key
                if api_key and api_key != 'None':
                    print 'Using existing API Key: %s' % api_key
                    key = {'api_key' : api_key}
                else:
                    key = cloud.account.create_key(email, password)
                    print 'API Key: %s' % key['api_key']
            
        else:
            key = cloud.account.get_key(email, password, api_key)
            print 'API Key: %s' % key['api_key']
        
        # save all key credentials
        if 'api_secretkey' in key:
            credentials.save_keydef(key)
        
        # set config and write it to file
        cloud.config.api_key = key['api_key']
        cloud.config.commit()
        cloud.cloudconfig.flush_config()        
                
        
        # if user is running "picloud setup" with sudo, we need to chown
        # the config file so that it's owned by user and not root.
        fix_sudo_path(os.path.join(cloud.cloudconfig.fullconfigpath,cloud.cloudconfig.configname))
        fix_sudo_path(cloud.cloudconfig.fullconfigpath)
        
        try:
            import platform
            conn = cloud._getcloudnetconnection()
            conn.send_request('report/install/', {'hostname': platform.node(),
                                                  'language_version': platform.python_version(),
                                                  'language_implementation': platform.python_implementation(),
                                                  'platform': platform.platform(),
                                                  'architecture': platform.machine(),
                                                  'processor': platform.processor(),
                                                  'pyexe_build' : platform.architecture()[0]
                                                  })
        except:
            pass
        
    except EOFError:
        sys.stderr.write('Got EOF. Please run "picloud setup" to complete installation.\n')
        sys.exit(1)
    except KeyboardInterrupt:
        sys.stderr.write('Got Keyboard Interrupt. Please run "picloud setup" to complete installation.\n')
        sys.exit(1)
    except cloud.CloudException, e:
        sys.stderr.write(str(e)+'\n')
        sys.exit(3)
    else:
        print '\nSetup successful!'
