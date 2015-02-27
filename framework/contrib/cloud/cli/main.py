#!/usr/bin/python
"""
Entry Point for the PiCloud Command-Line Interface (CLI)
"""
# since this module sits in the cloud package, we use absolute_import
# so that we can easily import the top-level package, rather than the
# cloud module inside the cloud package
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

import sys
import logging
import traceback

try:
    import json
except:
    # Python 2.5 compatibility
    import simplejson as json

import cloud
from UserDict import DictMixin

from . import argparsers
from .util import list_of_dicts_printer, dict_printer, list_printer,\
    key_val_printer, volume_ls_printer, cloud_info_printer,\
    cloud_result_printer, cloud_result_json_printer, bucket_info_printer,\
    no_newline_printer
from .setup_machine import setup_machine
from . import functions


def main(args=None):
    """Entry point for PiCloud CLI. If *args* is None, it is assumed that main()
    was called from the command-line, and sys.argv is used."""
    
    args = args or sys.argv[1:]
        
    # special case: we want to provide the full help information
    # if the cli is run with no arguments
    if len(args) == 0:
        argparsers.picloud_parser.print_help()
        sys.exit(1)
    
    # special case: if --version is specified at all, print it out
    if '--version' in args:
        print 'cloud %s' % cloud.__version__
        print 'running under python %s' % sys.version
        sys.exit(0)
        
    # parse_args is an object whose attributes are populated by the parsed args
    parsed_args = argparsers.picloud_parser.parse_args(args)
    module_name = getattr(parsed_args, '_module', '')
    command_name = getattr(parsed_args, '_command', '')
    function_name = module_name + ('.%s' % command_name if command_name else '')
        
    if function_name != 'setup':
        # using showhidden under setup will cause config to be flushed with hidden variables
        cloud.config._showhidden()
        cloud.config.verbose = parsed_args._verbose
        # suppress log messages
        cloud.config.print_log_level = logging.getLevelName(logging.CRITICAL)

    if parsed_args._api_key:
        cloud.config.api_key = parsed_args._api_key
    if parsed_args._api_secretkey:
        cloud.config.api_secretkey = parsed_args._api_secretkey
    if parsed_args._simulate:
        cloud.config.use_simulator = parsed_args._simulate     
    cloud.config.commit()
        
    # we take the attributes from the parsed_args object and pass them in
    # as **kwargs to the appropriate function. attributes with underscores
    # are special, and thus we filter them out.
    kwargs = dict([(k, v) for k,v in parsed_args._get_kwargs() if not k.startswith('_')])
                
    # handle post-op 
    for key, value in kwargs.items():
        if callable(value):
            kwargs[key] = value(**kwargs)

    
    # we keep function_mapping and printer_mapping here to prevent
    # circular imports
    
    # maps the output of the parser to what function should be called
    function_mapping = {'setup': setup_machine,
                        'exec': functions.execute,
                        'mapexec': functions.execute_map,
                        'status': functions.status,
                        'join': functions.join,
                        'result': functions.result,
                        'info': functions.info,
                        'kill': functions.kill,                        
                        'delete': functions.delete,
                        'ssh-info': cloud.shortcuts.ssh.get_ssh_info,
                        'ssh' : functions.ssh,
                        'exec-shell' : functions.exec_shell,
                        'rest.publish': functions.rest_publish, # move to rest?
                        'rest.remove' : cloud.rest.remove,
                        'rest.list' : cloud.rest.list,
                        'rest.info' : cloud.rest.info,
                        'rest.invoke' : functions.rest_invoke,
                        'rest.mapinvoke' : functions.rest_invoke_map,                                                
                        'files.get': cloud.files.get,
                        'files.put': cloud.files.put,
                        'files.list': cloud.files.list,
                        'files.delete': cloud.files.delete,
                        'files.get-md5': cloud.files.get_md5,
                        'files.sync-from-cloud': cloud.files.sync_from_cloud,
                        'files.sync-to-cloud': cloud.files.sync_to_cloud,
                        'bucket.get': cloud.bucket.get,
                        'bucket.put': cloud.bucket.put,
                        'bucket.iterlist': cloud.bucket.iterlist,
                        'bucket.list': cloud.bucket.list,
                        'bucket.info': cloud.bucket.info,
                        'bucket.remove': cloud.bucket.remove,
                        'bucket.remove-prefix': cloud.bucket.remove_prefix,
                        'bucket.get-md5': cloud.bucket.get_md5,
                        'bucket.sync-from-cloud': cloud.bucket.sync_from_cloud,
                        'bucket.sync-to-cloud': cloud.bucket.sync_to_cloud,
                        'bucket.make-public' : functions.bucket_make_public,
                        'bucket.make-private' : cloud.bucket.make_private,
                        'bucket.is-public' : cloud.bucket.is_public,
                        'bucket.public-url-folder' : cloud.bucket.public_url_folder,
                        'bucket.mpsafe-get' : cloud.bucket.mpsafe_get,
                        'realtime.request': cloud.realtime.request,
                        'realtime.release': cloud.realtime.release,
                        'realtime.list': cloud.realtime.list,
                        'volume.list': cloud.volume.get_list,
                        'volume.create': cloud.volume.create,
                        'volume.mkdir': cloud.volume.mkdir,
                        'volume.sync': cloud.volume.sync,
                        'volume.delete': cloud.volume.delete,
                        'volume.ls': cloud.volume.ls,
                        'volume.rm': cloud.volume.rm,
                        'env.list': cloud.environment.list_envs,
                        'env.list-bases': cloud.environment.list_bases,
                        'env.create': cloud.environment.create,
                        'env.edit-info': cloud.environment.edit_info,
                        'env.clone': cloud.environment.clone,
                        'env.modify': cloud.environment.modify,
                        'env.get-hostname': cloud.environment.get_setup_hostname,
                        'env.get-keypath': cloud.environment.get_key_path,
                        'env.save': cloud.environment.save,
                        'env.shutdown': cloud.environment.shutdown,
                        'env.save-shutdown': cloud.environment.save_shutdown,
                        'env.delete': cloud.environment.delete,
                        'env.ssh': cloud.environment.ssh,
                        'env.rsync': cloud.environment.rsync,
                        'env.run-script': cloud.environment.run_script,
                        #'queue.list': cloud.queue.list,
                        #'queue.create': cloud.queue.create,
                        #'queue.delete': cloud.queue.delete,
                        'cron.register': functions.cron_register, # move to rest?
                        'cron.deregister' : cloud.cron.deregister,
                        'cron.list' : cloud.cron.list,
                        'cron.run' : cloud.cron.manual_run,
                        'cron.info' : cloud.cron.info,
                        'wait-for.status' : cloud.wait_for.status,
                        'wait-for.port' : cloud.wait_for.port,
                        }
    
    # maps the called function to another function for printing the output
    printer_mapping = {'status' : key_val_printer('jid', 'status'),
                       'info' : cloud_info_printer,
                       'result' : cloud_result_printer,
                       'ssh-info' : dict_printer(['address', 'port', 'username', 'identity']),
                       'rest.list' : list_printer('label'),
                       'rest.info' : dict_printer(['label', 'uri', 'signature', 'output_encoding', 'description']),
                       'realtime.request': dict_printer(['request_id', 'type', 'cores', 'start_time']),
                       'realtime.list': list_of_dicts_printer(['request_id', 'type', 'cores', 'start_time']),
                       'files.list': list_printer('filename'),
                       'bucket.list': list_printer('filename'),
                       'bucket.iterlist': list_printer('filename'),
                       'bucket.info': bucket_info_printer,
                       'volume.list': list_of_dicts_printer(['name', 'mnt_path', 'created', 'desc']),
                       'volume.ls': volume_ls_printer,
                       'env.list': list_of_dicts_printer(['name', 'status', 'action', 'created', 'last_modified']),
                       'env.list-bases': list_of_dicts_printer(['name', 'distro', 'python_version']),
                       'env.get-hostname': no_newline_printer,
                       'env.get-keypath': no_newline_printer,
                       'env.ssh': no_newline_printer,
                       'env.run-script': no_newline_printer,
                       'queue.list' : list_printer('label'),
                       'cron.list' : list_printer('label'),
                       'cron.info' : dict_printer(['label', 'schedule', 'last_run', 'last_jid', 'created', 'creator_host', 'func_name']),
                       'wait-for.port' : dict_printer(['address', 'port']),
                       }
    
    json_printer_mapping = {'result' : cloud_result_json_printer}

    try:
        # execute function
        ret = function_mapping[function_name](**kwargs) 

        if parsed_args._output == 'json':
            # ordereddict issue.. fix once we on 2.7 only
            if (isinstance(ret, dict) or isinstance(ret, DictMixin)) and type(ret) != dict: 
                ret = dict(ret)
            json_printer = json_printer_mapping.get(function_name)
            if json_printer:
                json_printer(ret, parsed_args._output != 'no-header', kwargs)
            else:
                print json.dumps(ret)
        else:
            if function_name in printer_mapping:
                # use a printer_mapping if it exists
                # this is how dict/tables and lists with columns are printed
                printer_mapping[function_name](ret, parsed_args._output != 'no-header', kwargs)
            else:
                if isinstance(ret, (tuple, list)):
                    # if it's just  list with no mapping, just print it
                    for item in ret:
                        print item
                elif ret or isinstance(ret, bool):
                    # cases where the output is just a string or number or bool
                    print ret
                else:
                    # if the output is None, print nothing
                    pass
            
    except cloud.CloudException, e:
        if parsed_args._output == 'json':
            sys.stderr.write( json.dumps(e.args ) )
        else:
            # error thrown by cloud client library)
            sys.stderr.write(str(e)+'\n')
            sys.exit(3)
        
    except Exception, e:
        # unexpected errors
        sys.stderr.write('Got unexpected error\n')
        traceback.print_exc(file = sys.stderr)
        sys.exit(1)
        
    else:
        sys.exit(0)
