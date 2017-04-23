"""
PiCloud environment management.
This module allows the user to manage their environments on PiCloud.
See documentation at http://docs.picloud.com
"""
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
try:
    import json
except:
    # Python 2.5 compatibility
    import simplejson as json
import logging
import platform
import random
import re
import sys
import string
import time

import cloud
from .cloudlog import stdout as print_stdout, stderr as print_stderr
from .util import credentials
from .util import common

cloudLog = logging.getLogger('Cloud.environment')

plat = platform.system()

_urls = {'list': 'environment/list/',
         'list_bases': 'environment/list_bases/',
         'create': 'environment/create/',
         'edit_info': 'environment/edit_info/',
         'modify': 'environment/modify/',
         'save': 'environment/save/',
         'save_shutdown': 'environment/save_shutdown/',
         'shutdown': 'environment/shutdown/',
         'clone': 'environment/clone/',
         'delete': 'environment/delete/',
         }

# environment status types
_STATUS_CREATING = 'new'
_STATUS_READY = 'ready'
_STATUS_ERROR = 'error'

# environment action types
_ACTION_IDLE = 'idle'
_ACTION_SETUP = 'setup'
_ACTION_EDIT = 'edit'
_ACTION_SAVE = 'save'
_ACTION_SETUP_ERROR = 'setup_error'
_ACTION_SAVE_ERROR = 'save_error'


def _send_env_request(request_type, data, jsonize_values=True):
    type_url = _urls.get(request_type)
    if type_url is None:
        raise LookupError('Invalid env request type %s' % request_type)
    return common._send_request(type_url, data, jsonize_values)

"""
environment management
"""
def list_envs(name=None):
    """Returns a list of dictionaries describing user's environments.
    If *name* is given, only shows info for the environment with that name.

    Environment information is returned as list of dictionaries.  The keys
    within each returned dictionary are:

    * name: name of the environment
    * status: status of the environment
    * action: the action state of the environment (e.g. under edit)
    * created: time when the environment was created
    * last_modifed: last time a modification was saved
    * hostname: hostname of setup server if being modified
    * setup_time: time setup server has been up if being modified
    """
    resp = _send_env_request('list', {'env_name': name})
    return [common._fix_time_element(env, ['created', 'last_modified'])
            for env in resp['envs_list']]

def list_bases():
    """Returns a list of dictionaries describing available bases. The keys
    within each returned dictionary are:

    * id: id of the base (to be used when referencing bases in other functions)
    * name: brief descriptive name of the base
    """
    resp = _send_env_request('list_bases', {})
    return resp['bases_list']

def create(name, base, desc=None):
    """Creates a new cloud environment.

    * name: name of the new environment (max 30 chars)
    * base: name of the base OS to use for the environment (use list_bases to
        see list of bases and their names)
    * desc: Optional description of the environment (max 2000 chars)

    Returns the hostname of the setup server where the newly created
    environment can be modified.
    """
    pattern = '^[a-zA-Z0-9_-]*$'
    if not name:
        raise cloud.CloudException('No environment name given')
    elif len(name) > 30:
        raise cloud.CloudException('Environment name cannot be more than 30'
                                   ' characters')
    elif not re.match(pattern, name):
        raise cloud.CloudException('Environment name must consist of letters,'
                                   ' numbers, underscores, or hyphens')
    if desc and len(desc) > 2000:
        raise cloud.CloudException('Environment description cannot be more'
                                   ' than 2000 characters')

    resp = _send_env_request('create',
                             {'env_name': name, 'base_name': base,
                              'env_desc': desc or ''})
    cloudLog.debug('created environment %s', resp['env_name'])
    return get_setup_hostname(name)

def edit_info(name, new_name=None, new_desc=None):
    """Edits name and description of an existing environment.

    * name: current name of the environment
    * new_name: Optional new name of the environment (max 30 chars)
    * new_desc: Optional new description of the environment (max 2000 chars)
    """
    if new_name is None and new_desc is None:
        return

    pattern = '^[a-zA-Z0-9_-]*$'
    if not name:
        raise cloud.CloudException('No environment name given')

    if new_name is not None:
        if len(new_name) > 30:
            raise cloud.CloudException('Environment name cannot be more than 30'
                                       ' characters')
        elif not re.match(pattern, name):
            raise cloud.CloudException('Environment name must consist of letters,'
                                       ' numbers, underscores, or hyphens')
    if new_desc is not None and len(new_desc) > 2000:
        raise cloud.CloudException('Environment description cannot be more'
                                   ' than 2000 characters')

    resp = _send_env_request('edit_info',
                             {'old_env_name': name, 'new_name': new_name,
                              'new_desc': new_desc})
    cloudLog.debug('edited info for environment %s', resp['env_name'])

def modify(name):
    """Modifies an existing environment.

    * name: name of environment to modify

    Returns the hostname of the setup server where environment can be modified.
    """
    resp = _send_env_request('modify', {'env_name': name})
    cloudLog.debug('modify requested for env %s', resp['env_name'])
    return get_setup_hostname(name)

def save(name):
    """Saves the current modified version of the environment, without tearing
    down the setup server.
    
    * name: name of the environment to save
    
    This is a blocking function.  When it returns without errors, the new
    version of the environment is available for use by all workers.
    """
    resp = _send_env_request('save', {'env_name': name})
    cloudLog.debug('save requested for env %s', resp['env_name'])
    wait_for_edit(name)

def save_shutdown(name):
    """Saves the current modified version of the environment, and tears down
    the setup server when saving is done.
    
    * name: name of the environment to save
    
    This is a blocking function.  When it returns without errors, the new
    version of the environment is available for use by all workers.
    """
    resp = _send_env_request('save_shutdown', {'env_name': name})
    cloudLog.debug('save_shutdown requested for env %s', resp['env_name'])
    wait_for_idle(name)

def shutdown(name):
    """Tears down the setup server without saving the environment modification.
    
    * name: name of the environment to save
    """
    resp = _send_env_request('shutdown', {'env_name': name})
    cloudLog.debug('shutdown requested for env %s', resp['env_name'])
    wait_for_idle(name)

def clone(parent_name, new_name=None, new_desc=None):
    """Creates a new cloud environment by cloning an existing one.

    * parent_name: name of the existing environment to clone
    * new_name: new name of the environment. default is
        parent_name + "_clone". (max 30 chars)
    * new_desc: Optional description of the environment if different from
        parent environment description. (max 2000 chars)
    """
    pattern = '^[a-zA-Z0-9_-]*$'
    new_name = new_name or (parent_name + '_clone')
    if len(new_name) > 30:
        raise cloud.CloudException('Environment name cannot be more than 30'
                                   ' characters')
    elif not re.match(pattern, new_name):
        raise cloud.CloudException('Environment name must consist of letters,'
                                   ' numbers, underscores, or hyphens')
    if new_desc and len(new_desc) > 2000:
        raise cloud.CloudException('Environment description cannot be more'
                                   ' than 2000 characters')

    resp = _send_env_request('create',
                             {'parent_env_name': parent_name,
                              'env_name': new_name,
                              'env_desc': new_desc})
    cloudLog.debug('created environment %s', resp['env_name'])
    wait_for_idle(new_name)

def delete(name):
    """Deletes and existing environment.
    
    * name: Name of the environment to save
    """
    resp = _send_env_request('delete', {'env_name': name})
    cloudLog.debug('delete requested for env %s', resp['env_name'])

def get_setup_hostname(name):
    """Returns the hostname of the setup server where environment can be
    modified.  raises exception if the environment does not have a setup server
    already launched.
    
    * name: name of the environment whose setup server hostname is desired
    """
    env_info = wait_for_edit(name, _ACTION_IDLE)
    if env_info is None:
        raise cloud.CloudException('Environment is not being modified')
    return env_info['hostname']

def get_key_path():
    """Return the key file path for sshing into setup server."""
    api_key = cloud.connection_info().get('api_key')
    return credentials.get_sshkey_path(api_key)

def ssh(name, cmd=None):
    """Creates an ssh session to the environment setup server.
    
    * name: Name of the environment to make an ssh connection
    * cmd: By default, this function creates an interactive ssh session.
        If cmd is given, however, it executes the cmd on the setup server
        and returns the output of the command execution.
    """
    hostname = get_setup_hostname(name)
    key_path = get_key_path()
    status, stdout, stderr = common.ssh_session('picloud', hostname, key_path,
                                                run_cmd=cmd)
    if status:
        if stdout:
            sys.stdout.write(stdout)
        if stderr:
            sys.stderr.write(stderr)        
        sys.exit(status)

    if cmd and stdout:
        return stdout

def rsync(src_path, dest_path, delete=False, pipe_output=False):
    """Syncs data between a custom environment and the local filesystem. A
    setup server for the environment must already be launched. Also, keep in
    mind that the picloud user account (which is used for the rsync operation)
    has write permissions only to the home directory and /tmp on the setup
    server. If additional permissions are required, consider doing the rsync
    manually from the setup server using sudo, or rsync to the home directory
    then do a subsequent move using sudo.

    Either *src_path* or *dest_path* should specify an environment path, but
    not both. An environment path is of the format:

        env_name:[path-within-environment]

    Note that the colon is what indicates this is an environment path
    specification.
    
    *src_path* can be a list of paths, all of which should either be local
    paths, or environment paths. If *src_path* is a directory, a trailing slash
    indicates that its contents should be rsynced, while ommission of slash
    would lead to the directory itself being rsynced to the environment. 

    Example::

        rsync('~/dataset1', 'my_env:')

    will ensure that a directory named 'dataset1' will exist in the user
    picloud's home directory of environment 'my_env'. On the other hand,

        rsync(~/dataset1/', 'my_env:')

    will copy all the contents of 'dataset1' to the home directory of user
    picloud. See rsync manual for more information.
    
    If *delete* is True, files that exist in *dest_path* but not in *src_path*
    will be deleted.  By default, such files will not be removed.
    """
    conn = cloud._getcloudnetconnection()
    retry_attempts = conn.retry_attempts
    dest_is_local = common.is_local_path(dest_path)
    l_paths, r_paths = ((dest_path, src_path) if dest_is_local else
                        (src_path, dest_path))
    local_paths = common.parse_local_paths(l_paths)
    env_name, env_paths = common.parse_remote_paths(r_paths)
    
    hostname = get_setup_hostname(env_name)
    try:
        r_base = 'picloud@%s:' % hostname
        r_paths = ' '.join(['%s%s' % (r_base, path) for path in env_paths])
        l_paths = ' '.join(local_paths)
        sync_args = (r_paths, l_paths) if dest_is_local else (l_paths, r_paths)

        for attempt in xrange(retry_attempts):
            exit_code, _, _ = common.rsync_session(*sync_args, delete=delete,
                                                   pipe_output=pipe_output)
            if not exit_code:
                break
            print_stderr('Retrying environment rsync...')
        else:
            raise Exception('rsync failed multiple attempts... '
                            'Please contact PiCloud support')
    except Exception as e:
        cloudLog.error('Environment rsync errored with:\n%s', e)
        print e

def run_script(name, filename):
    """Runs a script on the environment setup server, and returns the output.
    
    * name: Environment whose setup server should run the script
    filename: local path where the script to be run can be found
    """
    POPU = string.ascii_letters + string.digits
    dest_file = ''.join(random.sample(POPU, 16))
    try:
        rsync(filename, '%s:%s' % (name, dest_file), pipe_output=True)
        run = "chmod 700 {0}; ./{0} &> {0}.out; cat {0}.out".format(dest_file)
        output = ssh(name, run)
    except Exception as e:
        cloudLog.error('script could not be run: %s', str(e))
        print 'Script could not be run on the setup server.'
        print e
    else:
        return output
    finally:
        ssh(name, "rm -rf %s*" % dest_file)

def wait_for_idle(name, invalid_actions=None):
    """Waits for environment to be in idle action state."""
    return _wait_for(name=name, action=_ACTION_IDLE,
                     invalid_actions=invalid_actions)

def wait_for_edit(name, invalid_actions=None):
    """Waits for environment to be in edit action state."""
    return _wait_for(name=name, action=_ACTION_EDIT,
                     invalid_actions=invalid_actions)

def _wait_for(name, action, invalid_actions=None, poll_frequency=2,
              max_poll_duration=1800):
    """Generic wait function for polling until environment reaches the
    specified action state. Raises exception if the environment ever falls into
    an error status or action state.
    """
    invalid_actions = invalid_actions or []
    if not hasattr(invalid_actions, '__iter__'):
        invalid_actions = [invalid_actions]
    
    for _ in xrange(max_poll_duration / poll_frequency):
        resp = list_envs(name)

        if len(resp) == 0:
            raise cloud.CloudException('No matching environment found.')
        elif len(resp) != 1:
            cloudLog.error('single env query returned %s results', len(resp))
            raise cloud.CloudException('Unexpected result from PiCloud. '
                                       'Please contact PiCloud support.')
        env_info = resp.pop()
        resp_status = env_info['status']
        resp_action = env_info['action']
        if resp_status == _STATUS_ERROR:
            raise cloud.CloudException('Environment creation failed. '
                                       'Please contact PiCloud support.')
        elif resp_status == _STATUS_READY:
            if resp_action == _ACTION_SETUP_ERROR:
                raise cloud.CloudException('Setup server launch failed. '
                                           'Please contact PiCloud support.')
            elif resp_action == _ACTION_SAVE_ERROR:
                raise cloud.CloudException('Environment save failed. '
                                           'Please contact PiCloud support.')
            elif resp_action in invalid_actions:
                return None
            elif resp_status == _STATUS_READY and action == resp_action:
                return env_info
        elif resp_status == _STATUS_CREATING:
            pass

        time.sleep(poll_frequency)
        
    raise cloud.CloudException('Environment operation timed out. '
                               'Please contact PiCloud support.')
