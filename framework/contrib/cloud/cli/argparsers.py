"""
ArgumentParsers for processing input to the PiCloud CLI.
"""
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

import os
import errno

try:
    import argparse
except:
    from . import argparse


def shell_path_post_op(path):
    """Makes *path* specification in the CLI more natural for shell usage.
    
    Converts "." to "name"
    Converts "existing_dir/" to "existing_dir/name"
    
    Usage: put shell_path_post_op as a type when specifying a new argument
    for a parser"""
    
    def fix_path(**kwargs):
        
        # extract key_name 
        name = kwargs.get('name')
        if not name:
            name = kwargs.get('key_name')
        if not name:
            name = kwargs.get('obj_path')
        if not name:
            raise ValueError('no (key)name specified') 
        
        if os.path.isdir(path):
            return os.path.join(path, name)
        else: # dir may not exist yet
            dname = os.path.dirname(path)
            if dname:
                try:
                    os.makedirs(dname)
                except OSError, e:
                    if e.errno != errno.EEXIST:
                        raise
            basename = os.path.basename(path)
            if basename:  # dest is a file
                return path
            else:
                return os.path.join(path, name)
            
    return fix_path


def common_arg(*args, **kwargs):
    return args, kwargs


"""Primary Parser"""
picloud_parser = argparse.ArgumentParser(prog='picloud', description='PiCloud Command-Line Interface (CLI)')
picloud_parser.add_argument('--version', dest='_version', action='store_true', help='Print version')
picloud_parser.add_argument('-v', '--verbose', dest='_verbose', action='store_true', help='Increase amount of information you are given during command execution')
picloud_parser.add_argument('-o', '--output', dest='_output', default='default', choices=['json', 'no-header', 'default'], help='Format of output')
picloud_parser.add_argument('-a', '--api-key', dest='_api_key', type=int, help='API key to use')
picloud_parser.add_argument('-k', '--api-secretkey', dest='_api_secretkey', help='API secret key that matches the API key')
picloud_parser.add_argument('-s', '--simulate', dest='_simulate', action='store_true', help='Run command in simulator (see simulator documentation)')

picloud_subparsers = picloud_parser.add_subparsers(title='subcommands',
                                                   description='Subcommands and submodules of the PiCloud CLI. For detailed help, use "-h" after the subcommand/submodule. For example, for assistance with "exec", use "picloud exec -h".',
                                                   dest='_module',
                                                   )

"""Setup Parser"""
setup_parser = picloud_subparsers.add_parser('setup', description='Sets up the current machine to use PiCloud.',
                                             help='Set up the current machine to use PiCloud')
setup_parser.add_argument('--email', '-e', help='Email used to login to your PiCloud account')
setup_parser.add_argument('--password', '-p', help='Password used to login to your PiCloud account')
setup_parser.add_argument('--api-key', '-a', nargs='?', default=False, help='API Key to use. If specified without a value, a new API Key will be created without prompting')

"""Execute Parser"""
jid_help_str = "jid or comma-seperated list of jids. Ranges may be specified with '-', e.g. 1-4 for jobs 1,2, and 3"

# args common to call/map/rest-reg/etc
# We can't use parents as want these to appear AFTER data argument
common_args = [
    common_arg('-t', '--type', default='c1', choices=['c1', 'c2', 'f2', 'm1', 's1'], 
               help='Type of compute resource to use. Default c1'),
    common_arg('-r', '--return', metavar='FILENAME', dest='return_file',  
               help='If provided result of job will be contents of FILENAME. If not set, result is stdout'),
    common_arg('-e', '--env', help='Name of the environment to run job within'),
    common_arg('-v', '--vol', action='append', help='Provide job with access to this volume'),
    common_arg('-l', '--label', help='Label to attach to job. Allows for job filtering'),    
    common_arg('-c', '--cores', default=1, type=int, choices=[1, 2, 4, 8, 16], 
               help='Number of cores for your job to utilize'),
    common_arg('-i', '--ignore-exit-status', action='store_true',
               help='If provided, job will not error if command exits with a nonzero error code'),               
    common_arg('-w', '--cwd', metavar='DIRECTORY', help='Directory to execute command within'),
    common_arg('-m', '--max-runtime', type=int, help='Job will be terminated if it runs beyond integer MAX_RUNTIME minutes'),
    common_arg('-x', '--depends-on', metavar='JIDS', help='Jobs this depends on. JIDS are a %s' % jid_help_str),
    common_arg('--depends-on-errors', default='abort', choices=['abort', 'ignore'], 
               help='Policy for how a jid in depends-on erroring should be handled. If abort (default), set job to stalled. If ignore, treat error as satisfying'),
    common_arg('--not-restartable', dest='restartable', action='store_false',
               help='Indicates that job cannot be safely restarted on a hardware failure'),
    common_arg('--priority', type=int,
               help="Positive integer describing job's priority. PiCloud will try to run jobs with lower priorities earlier")

]               

exec_parser = picloud_subparsers.add_parser('exec', description='Executes a program on PiCloud through the shell. This is the shell version of cloud.call',
                                            help="Execute a program on PiCloud through the shell")
exec_parser.add_argument('command', nargs=argparse.PARSER, help='Templated shell command to execute')
exec_parser.add_argument('-d', '--data', dest='args', metavar='PARAMETER=VALUE', action='append', 
                         help='Set template parameter to value')
for args, kwargs in common_args:
    exec_parser.add_argument(*args, **kwargs)

mapexec_parser = picloud_subparsers.add_parser('mapexec', description='Executes many programs in parallel on PiCloud through the shell. This is the shell version of cloud.map. \
Number of programs will be determined by number of comma-seperated arguments provided to -n (--maparg)',
                                            help="Execute parallel programs on PiCloud through the shell")
mapexec_parser.add_argument('command', nargs=argparse.PARSER, help='Templated Shell command to execute.')
mapexec_parser.add_argument('-d', '--data', dest='args', metavar='PARAMETER=VALUE', action='append', 
                         help='For every mapjob, set template parameter to value')
mapexec_parser.add_argument('-n', '--map-data', dest='mapargs', metavar='PARAMETER=VALUE1,VALUE2,..', action='append',  
                         help="Specify values for template parameters. job1 parameter takes value1, job2 takes value2, etc.")
mapexec_parser.add_argument('-N', '--arg-file', dest='argfiles', metavar='PARAMETER=FILENAME', action='append',
                         help="Specify a file that specifies template parameter values. Each line corresponds to the argument for one job.")
mapexec_parser.add_argument('-k', '--copies', dest='duplicates', metavar='NUM_DUPLICATES', default=1, type=int,  
                        help='Duplicate jobs. Have NUM_DUPLICATES jobs take a given parameter value'),


for args, kwargs in common_args:
    mapexec_parser.add_argument(*args, **kwargs)


"""Additional PiCloud commands"""




jid_parser = argparse.ArgumentParser(add_help=False)
jid_parser.add_argument('jids', help=jid_help_str) 

status_parser = picloud_subparsers.add_parser('status', description='Obtain status of job(s)', 
                                              parents=[jid_parser], help='Status of job(s)')

result_parser = picloud_subparsers.add_parser('result', description='Obtain result of job(s)', 
                                              parents=[jid_parser], help='Result of job(s)')
result_parser.add_argument('-t', '--timeout', metavar='SECONDS', default=None, type=float,
                           help='Error if job has not finished by this number of seconds')
result_parser.add_argument('-i', '--ignore-errors', action='store_true', default=False,
                           help='If job errored print Exception as return value and exit 0 rather than aborting with Exception')


join_parser = picloud_subparsers.add_parser('join', description='Wait until job(s) running on PiCloud are complete.', 
                                              parents=[jid_parser], help='Block until job(s) finished')
join_parser.add_argument('-t', '--timeout', metavar='SECONDS', default=None, type=float,
                         help='Error if job has not finished by this number of seconds')

info_parser = picloud_subparsers.add_parser('info', description='Obtain information about job(s)', 
                                            parents=[jid_parser], help='Information about job(s)')
info_parser.add_argument('-o', '--output', dest='info_requested', default=None, 
                         help='comma seperated list of info desired. See docs for full listing. e.g. status, runtime')

kill_parser = picloud_subparsers.add_parser('kill', description='Abort running of job(s)',
                                            parents=[jid_parser], help='Kill job(s)')

delete_parser = picloud_subparsers.add_parser('delete', description='Delete job(s) from PiCloud', 
                                              parents=[jid_parser], help='Delete job(s)')

ssh_parser = picloud_subparsers.add_parser('ssh', description='Wait until job is processing and ssh into the container of a job',
                                           help='SSH into a job')
ssh_parser.add_argument('-t', '--timeout', metavar='SECONDS', default=None, type=float,
                        help="Error if ssh not ready by this number of seconds.")
ssh_parser.add_argument('jid', type=int, help="job identifier")
ssh_parser.add_argument('cmd', nargs=argparse.REMAINDER, help='Optional command to run without opening interactive console (command should be usually be escaped with quotes)') 


ssh_info_parser = picloud_subparsers.add_parser('ssh-info', description='Obtain IP Address, Port, and username info to connect to a job for ssh/scp/etc. Use picloud ssh to log in to a job',
                                                help='Obtain SSH information')
ssh_info_parser.add_argument('-t', '--timeout', metavar='SECONDS', default=None, type=float,
                             help="Error if ssh not not ready by this number of seconds.")
ssh_info_parser.add_argument('jid', type=int, help="job identifier") 


exec_shell_parser = picloud_subparsers.add_parser('exec-shell', description='Test your configuration by starting a session defined by your environment, volumes, etc. and logging into it',
                                            help="SSH into configuration")
exec_shell_parser.add_argument('--timeout', metavar='SECONDS', default=None, type=float,
                             help="Error if ssh not not ready by this number of seconds.")
exec_shell_parser.add_argument('-k','--keep-alive', action='store_true', help='Keep SSH job alive after all ssh connections terminated')
exec_shell_parser.add_argument('-t', '--type', default='c1', choices=['c1', 'c2', 'f2', 'm1', 's1'], 
               help='Type of compute resource to use. Default c1')
exec_shell_parser.add_argument('-e', '--env', help='Name of the environment to run job within')
exec_shell_parser.add_argument('-v', '--vol', action='append', help='Provide job with access to this volume')    
exec_shell_parser.add_argument('-c', '--cores', default=1, type=int, choices=[1, 2, 4, 8, 16], 
               help='Number of cores for your job to utilize')
exec_shell_parser.add_argument('-m', '--max-runtime', type=int, help='Job will be terminated if it runs beyond integer MAX_RUNTIME minutes')


"""Rest Parser"""
rest_parser = picloud_subparsers.add_parser('rest', description="Module for managing PiCloud REST interfaces", 
                                             help="Manage REST Interfaces")
rest_subparsers = rest_parser.add_subparsers(title='commands', dest='_command', help='command help')
rest_publish_parser = rest_subparsers.add_parser('publish', help='Publish a shell command to PiCloud which can be executed over REST')
rest_publish_parser.add_argument('label', help='Label to assign the published function')
rest_publish_parser.add_argument('command', nargs=argparse.PARSER, help='Templated shell command to execute')
for args, kwargs in common_args:
    dest = kwargs.get('dest')
    if dest in ['depends_on', 'depends_on_errors', 'label']: # not relevant to rest publish
        continue
    rest_publish_parser.add_argument(*args, **kwargs)

rest_list_parser = rest_subparsers.add_parser('list', help='List functions published to PiCloud')

rest_info_parser = rest_subparsers.add_parser('info', help='Retrieve information about a published function')
rest_info_parser.add_argument('label', help='Label of published function to get info about')

rest_invoke_parser = rest_subparsers.add_parser('invoke', help='Invoke a function published on PiCloud')
rest_invoke_parser.add_argument('label', help='Label of function to invoke')
rest_invoke_parser.add_argument('-d', '--data', dest='args', metavar='PARAMETER=VALUE', action='append', 
                         help='Set template parameter to value')

rest_mapinvoke_parser = rest_subparsers.add_parser('mapinvoke', help='Invoke a function published on PiCloud many times')
rest_mapinvoke_parser.add_argument('label', help='Label of function to invoke')
rest_mapinvoke_parser.add_argument('-d', '--data', dest='args', metavar='PARAMETER=VALUE', action='append', 
                         help='For every mapjob, set template parameter to value')
rest_mapinvoke_parser.add_argument('-n', '--map-data', dest='mapargs', metavar='PARAMETER=VALUE1,VALUE2,..', action='append',  
                         help="Specify template parameter for each value. e.g. job1 parameter takes value1, job2 takes value2, etc.")


rest_remove_parser = rest_subparsers.add_parser('remove', help='Remove a published function from PiCloud')
rest_remove_parser.add_argument('label', help='Label of published function to remove')


"""Files Parser"""
files_parser = picloud_subparsers.add_parser('files', description="Module for managing files stored on PiCloud's key-value store (Deprecated: see picloud bucket)", 
                                             help="(Deprecated: see bucket) Manage files on key-value store")
files_subparsers = files_parser.add_subparsers(title='commands', dest='_command', help='command help')

files_delete_parser = files_subparsers.add_parser('delete', help='Delete a file stored on PiCloud')
files_delete_parser.add_argument('name', default=None, help='Name of file stored on PiCloud')

files_get_parser = files_subparsers.add_parser('get', help='Retrieve a file from PiCloud')
files_get_parser.add_argument('name', help='Name of file in storage')
files_get_parser.add_argument('destination', type=shell_path_post_op, help='Local path to save file to')
#files_get_parser.add_argument('--start-byte', help='Starting byte')
#files_get_parser.add_argument('--end-byte', help='Ending byte')

files_getmd5_parser = files_subparsers.add_parser('get-md5', help='Get the md5 checksum of a file stored on PiCloud')
files_getmd5_parser.add_argument('name', default=None, help='Name of file stored on PiCloud')

files_list_parser = files_subparsers.add_parser('list', help='List files in PiCloud Storage')

files_put_parser = files_subparsers.add_parser('put', help='Store a file on PiCloud')
files_put_parser.add_argument('source', help='Local path to file')
files_put_parser.add_argument('name', default=None, help='Name for file to be stored on PiCloud')

files_syncfromcloud_parser = files_subparsers.add_parser('sync-from-cloud', help='Download file if it does not exist locally or has changed')
files_syncfromcloud_parser.add_argument('name', default=None, help='Name of file stored on PiCloud')
files_syncfromcloud_parser.add_argument('destination', type=shell_path_post_op, help='Local path to save file to')

files_synctocloud_parser = files_subparsers.add_parser('sync-to-cloud', help='Upload file only if it does not exist on PiCloud or has changed')
files_synctocloud_parser.add_argument('source', default=None, help='local path to file')
files_synctocloud_parser.add_argument('name', default=None, help='Name for file to be stored on PiCloud')

"""Bucket Parser"""
bucket_parser = picloud_subparsers.add_parser('bucket', description="Module for managing bucket objects stored on PiCloud's key-value store",
                                              help="Manage data objects in your bucket")
bucket_subparsers = bucket_parser.add_subparsers(title='commands', dest='_command', help='command help')

bucket_remove_parser = bucket_subparsers.add_parser('remove', help='Remove a bucket object from PiCloud')
bucket_remove_parser.add_argument('obj_paths', metavar='obj-path', default=None, help='Name of bucket object')
bucket_remove_parser.add_argument('-p', '--prefix', default=None, type=str, help='If provided, prepend PREFIX/ to obj-path')

bucket_remove_prefix_parser = bucket_subparsers.add_parser('remove-prefix', help='Remove list of bucket objects from PiCloud beginnign with prefix')
bucket_remove_prefix_parser.add_argument('prefix', default=None, help='Remove all objects beginning witht his prefix')


bucket_get_parser = bucket_subparsers.add_parser('get', help='Retrieve a bucket object from PiCloud')
bucket_get_parser.add_argument('obj_path', metavar='obj-path', help='Name of bucket object')
bucket_get_parser.add_argument('file_path', metavar='file-path', type=shell_path_post_op, help='Local path to save bucket object to')
bucket_get_parser.add_argument('-p', '--prefix', default=None, type=str, help='If provided, prepend PREFIX/ to obj-path')
bucket_get_parser.add_argument('--start-byte', default=0, type=int, help='Starting byte')
bucket_get_parser.add_argument('--end-byte', default=None, type=int,help='Ending byte')


bucket_getmd5_parser = bucket_subparsers.add_parser('get-md5', help='Get the md5 checksum of a bucket object stored on PiCloud')
bucket_getmd5_parser.add_argument('obj_path', metavar='obj-path', default=None, help='Name of bucket object')
bucket_getmd5_parser.add_argument('-p', '--prefix', default=None, type=str, help='If provided, prepend PREFIX/ to obj-path')

bucket_iterlist_parser = bucket_subparsers.add_parser('iterlist', help='List bucket object keys stored on PiCloud (complete listing)')
bucket_iterlist_parser.add_argument('-f', '--folderize', action="store_true", default=False,  
                                    help='Treat listing as directory based; compact keys containing "/" into a single folder')
bucket_iterlist_parser.add_argument('-p', '--prefix', default=None, type=str, 
                                    help='Return only keys beginning with PREFIX. If folderize, list a folder by setting PREFIX to the folder')

bucket_list_parser = bucket_subparsers.add_parser('list', help='List bucket object keys stored on PiCloud (result may be incomplete)')
bucket_list_parser.add_argument('-f', '--folderize', action="store_true", default=False,  
                                help='Treat listing as directory based; compact keys containing "/" into a single folder')
bucket_list_parser.add_argument('-p', '--prefix', default=None, type=str, 
                                help='Return only keys beginning with PREFIX. If folderize, list a folder by setting PREFIX to the folder')
bucket_list_parser.add_argument('-m', '--marker', default=None, type=str, 
                                help='Return only keys where key > marker')
bucket_list_parser.add_argument('-k', '--max-keys', default=1000, type=int, 
                                help='Maxmimum number of names that can be returned (max 1000)')

bucket_info_parser = bucket_subparsers.add_parser('info', help='Return information about bucket object')
bucket_info_parser.add_argument('obj_path', metavar='obj-path', default=None, help='Name of bucket object')
bucket_info_parser.add_argument('-p', '--prefix', default=None, type=str,
                                help='If provided, prepend PREFIX/ to obj-path')

bucket_put_parser = bucket_subparsers.add_parser('put', help='Store a bucket object on PiCloud')
bucket_put_parser.add_argument('file_path', metavar='file-path', help='Local path to upload from')
bucket_put_parser.add_argument('obj_path', metavar='obj-path', default=None, help='Key (name) of the bucket object to be stored on PiCloud')
bucket_put_parser.add_argument('-p', '--prefix', default=None, type=str,
                               help='If provided, prepend PREFIX/ to obj-path')

bucket_syncfromcloud_parser = bucket_subparsers.add_parser('sync-from-cloud', help='Download bucket object only if it has changed')
bucket_syncfromcloud_parser.add_argument('obj_path', metavar='obj-path', default=None, help='Name of bucket object in storage')
bucket_syncfromcloud_parser.add_argument('file_path', metavar='file-path', type=shell_path_post_op, help='Local path to save bucket object to')
bucket_syncfromcloud_parser.add_argument('-p', '--prefix', default=None, type=str,
                                         help='If provided, prepend PREFIX/ to obj-path')


bucket_synctocloud_parser = bucket_subparsers.add_parser('sync-to-cloud', help='Upload bucket object only if it has changed')
bucket_synctocloud_parser.add_argument('file_path', metavar='file-path', default=None, help='Local path to upload from')
bucket_synctocloud_parser.add_argument('obj_path', metavar='obj-path', default=None, help='Name for bucket object to be stored on PiCloud')
bucket_synctocloud_parser.add_argument('-p', '--prefix', default=None, type=str,
                                       help='If provided, prepend PREFIX/ to obj-path')


bucket_make_public_parser = bucket_subparsers.add_parser('make-public', help='Makes the bucket object publicly accessible by a URL')
bucket_make_public_parser.add_argument('obj_path', metavar='obj-path', default=None, help='Name of bucket object in storage')
bucket_make_public_parser.add_argument('-p', '--prefix', default=None, type=str,
                                       help='If provided, prepend PREFIX/ to obj-path')
bucket_make_public_parser.add_argument('-d', '--header', dest='header_args', metavar='PARAMETER=VALUE', action='append', 
                                       help='Customize headers that should in HTTP response. See Python docs on cloud.bucket.make_public')
bucket_make_public_parser.add_argument('-r', '--reset-headers', action='store_true', default=False,
                                       help='Clear all custom HTTP Headers')

bucket_make_private_parser = bucket_subparsers.add_parser('make-private', help='Removes the publicly accessible URL associated with the bucket object')
bucket_make_private_parser.add_argument('obj_path', metavar='obj-path', default=None, help='Name of bucket object in storage')
bucket_make_private_parser.add_argument('-p', '--prefix', default=None, type=str,
                                        help='If provided, prepend PREFIX/ to obj-path')

bucket_is_public = bucket_subparsers.add_parser('is-public', help='Determine if bucket object is publicly accessible by a URL')
bucket_is_public.add_argument('obj_path', metavar='obj-path', default=None, help='Name of bucket object in storage')
bucket_is_public.add_argument('-p', '--prefix', default=None, type=str,
                              help='If provided, prepend PREFIX/ to obj-path')

bucket_public_url_folder = bucket_subparsers.add_parser('public-url-folder', help='Return HTTP path that begins all your public bucket URLs')

bucket_mpsafe_get_parser = bucket_subparsers.add_parser('mpsafe-get', help='Atomically retrieve a bucket object from PiCloud. Use this in lieu of get/sync when multiple processes need to get a bucket')
bucket_mpsafe_get_parser.add_argument('obj_path', metavar='obj-path', help='Name of bucket object')
bucket_mpsafe_get_parser.add_argument('file_path', metavar='file-path', type=shell_path_post_op, help='Local path to save bucket object to')
bucket_mpsafe_get_parser.add_argument('-p', '--prefix', default=None, type=str, help='If provided, prepend PREFIX/ to obj-path')
bucket_mpsafe_get_parser.add_argument('--start-byte', default=0, type=int, help='Starting byte')
bucket_mpsafe_get_parser.add_argument('-t', '--timeout', metavar='SECONDS', default=None, type=float,
                           help="Error if timeout not acquired by by this number of seconds.")
bucket_mpsafe_get_parser.add_argument('-s', '--sync', dest='do_sync', action='store_true',
                               help='If set, obtain file if local is different from version on PiCloud; if not set, only obtain file it does not exist locally')


"""Realtime Parser"""
realtime_parser = picloud_subparsers.add_parser('realtime', description='Module for managing realtime cores.', help = "Manage realtime cores")
realtime_subparsers = realtime_parser.add_subparsers(title='commands', dest='_command', help='command help')

realtime_list_parser = realtime_subparsers.add_parser('list', help='List realtime reservations')

realtime_release_parser = realtime_subparsers.add_parser('release', help='Release realtime cores')
realtime_release_parser.add_argument('request-id')

realtime_request_parser = realtime_subparsers.add_parser('request', help='Request realtime cores')
realtime_request_parser.add_argument('-m', '--max-duration', metavar='HOURS', default=None, type=int,
                                     help='Time for request to be active.')
realtime_request_parser.add_argument('type', choices=['c1', 'c2', 'f2', 'm1', 's1'], help='The type of core to reserve.')
realtime_request_parser.add_argument('cores', type=int, help='The number of cores to reserve.')

""" Volume Parser """
volume_parser = picloud_subparsers.add_parser('volume', description='Module for managing volumes stored on PiCloud', help = "Manage volumes")
volume_subparsers = volume_parser.add_subparsers(title='commands', dest='_command', help='command help')

volume_list_parser = volume_subparsers.add_parser('list', help='List existing volumes', description='Lists existing cloud volumes')
volume_list_parser.add_argument('-n', '--name', nargs='+', help='Name(s) of volume to list')
volume_list_parser.add_argument('-d', '--desc', action='store_true', default=False, help='Print volume description')

volume_create_parser = volume_subparsers.add_parser('create', help='Create a volume on PiCloud')
volume_create_parser.add_argument('name', help='Name of the volume to create (max 64 chars)')
volume_create_parser.add_argument('mount_path', metavar='mount-path', help='Mount point (is relative then relative to /home/picloud) where jobs should expect this volume.')
volume_create_parser.add_argument('-d', '--desc', default=None, help='Description of the volume (max 1024 chars)')

volume_mkdir_parser = volume_subparsers.add_parser('mkdir', help='Create directory(ies) at cloud volume [path]')
volume_mkdir_parser.add_argument('volume_path', metavar='volume-path', nargs='+', help='Cloud volume path where directory should be created')
volume_mkdir_parser.add_argument('-p', '--parents', action='store_true', default=False, help='Make necessary parents')

volume_sync_parser = volume_subparsers.add_parser('sync', help='Sync local directory to volume on PiCloud', description='Syncs a local path and a cloud volume.', formatter_class=argparse.RawDescriptionHelpFormatter,)
volume_sync_parser.add_argument('source', nargs='+', help='Source path that should be synced')
volume_sync_parser.add_argument('dest', help='Destination path that should be synced')
volume_sync_parser.add_argument('-d', '--delete', action='store_true', help='Delete destination files that do not exist in source')

volume_delete_parser = volume_subparsers.add_parser('delete', help='Delete a cloud volume')
volume_delete_parser.add_argument('name', help='Name of the cloud volume to delete')

volume_ls_parser = volume_subparsers.add_parser('ls', help='List the contents of a cloud volume [path]')
volume_ls_parser.add_argument('volume_path', metavar='volume-path', nargs='+', help='Cloud volume path whose contents should be shown')
volume_ls_parser.add_argument('-l', '--extended-info', action='store_true', default=False, help='Use long listing format')

volume_rm_parser = volume_subparsers.add_parser('rm', help='Remove contents from a cloud volume')
volume_rm_parser.add_argument('volume_path', metavar='volume-path', nargs='+', help='Cloud volume path whose contents should be removed')
volume_rm_parser.add_argument('-r', '--recursive', action='store_true', default=False, help='Remove directories and their contents recursively')

""" Environment Parser """
environment_parser = picloud_subparsers.add_parser('env', description='Module for managing custom environments', help = "Manage environments")
environment_subparsers = environment_parser.add_subparsers(title='commands', dest='_command', help='command help')

environment_list_parser = environment_subparsers.add_parser('list', help='List existing environments', description='Lists existing custom environments')
environment_list_parser.add_argument('-n', '--name', nargs='+', help='Name(s) of environments to list')

environment_list_bases_parser = environment_subparsers.add_parser('list-bases', help='List base environments', description='Lists base environments from which to build custom environments')

environment_create_parser = environment_subparsers.add_parser('create', help='Create a new custom environment')
environment_create_parser.add_argument('name', help='Name of the environment to create (max 30 chars)')
environment_create_parser.add_argument('base', help='Name of the base OS to use (use "env list_bases" for base names)')
environment_create_parser.add_argument('-d', '--desc', default=None, help='Description of the environment (max 2000 chars)')

environment_clone_parser = environment_subparsers.add_parser('clone', help='Clone an existing custom environment')
environment_clone_parser.add_argument('parent_name', metavar='parent-name', help='Name of the environment to clone')
environment_clone_parser.add_argument('new_name', metavar='new-name', nargs='?', default=None, help='Name of cloned environment (default adds _clone to parent name)')
environment_clone_parser.add_argument('-d', '--new-desc', default=None, help='Description of the environment (max 2000 chars)')

environment_modify_parser = environment_subparsers.add_parser('modify', help='Modify and existing environment')
environment_modify_parser.add_argument('name', help='Name of existing environment to modify')

environment_get_hostname_parser = environment_subparsers.add_parser('get-hostname', help='Get the hostname of environment setup server')
environment_get_hostname_parser.add_argument('name', help='Name of the environment to query for setup server hostname')

environment_get_keypath_parser = environment_subparsers.add_parser('get-keypath', help='Get the path of the ssh key for setup servers')

environment_save_parser = environment_subparsers.add_parser('save', help='Save the current modifications for an environment without shutting down setup server')
environment_save_parser.add_argument('name', help='Name of the environment to save')

environment_shutdown_parser = environment_subparsers.add_parser('shutdown', help='Shutdown the setup server for an environment without saving changes')
environment_shutdown_parser.add_argument('name', help='Name of the environment to shutdown')

environment_save_shutdown_parser = environment_subparsers.add_parser('save-shutdown', help='Save the current modifications for an environment and shutdown the setup server')
environment_save_shutdown_parser.add_argument('name', help='Name of the environment to save and shutdown')

environment_delete_parser = environment_subparsers.add_parser('delete', help='Delete an environment')
environment_delete_parser.add_argument('name', help='Name of the environment to delete')

environment_edit_info_parser = environment_subparsers.add_parser('edit-info', help='Edit name and description of an environment')
environment_edit_info_parser.add_argument('name', help='Name of the environment to edit.')
environment_edit_info_parser.add_argument('-n', '--new-name', default=None, help='New name of the environment (max 30 chars)')
environment_edit_info_parser.add_argument('-d', '--new-desc', default=None, help='New description of the environment (max 2000 chars)')

environment_ssh_parser = environment_subparsers.add_parser('ssh', help='Connect to an environment via ssh')
environment_ssh_parser.add_argument('name', help='Name of the environment to connect')
environment_ssh_parser.add_argument('cmd', nargs=argparse.REMAINDER, help='Optional command to run without opening interactive console (command should be usually be escaped with quotes)')

environment_rsync_parser = environment_subparsers.add_parser('rsync', help='rsync to environment setup server')
environment_rsync_parser.add_argument('src_path', metavar='src-path', nargs='+', help='Source path to rsync')
environment_rsync_parser.add_argument('dest_path', metavar='dest-path', help='Destination path to rsync')
environment_rsync_parser.add_argument('-d', '--delete', action='store_true', help='Delete destination files that do not exist in source')

environment_run_script_parser = environment_subparsers.add_parser('run-script', help='Run a script on the environment setup server')
environment_run_script_parser.add_argument('name', help='Name of the environment to run the script')
environment_run_script_parser.add_argument('filename', help='Local path to script to run on the environment setup server')


""" Queue Parser """

"""
queue_parser = picloud_subparsers.add_parser('queue', description="Module for managing PiCloud Queues", 
                                             help="Manage queues")
queue_parsers = queue_parser.add_subparsers(title='commands', dest='_command', help='command help')

queue_attach_parser = queue_parsers.add_parser('attach', help='Attach a command to process queue messages')
queue_attach_parser.add_argument('name', help='Name of queue')
queue_attach_parser.add_argument('messager_handler', metavar='message-handler', help='Crontab schedule. See documentation for format details')
queue_attach_parser.add_argument('command', nargs=argparse.PARSER, help='Templated shell command to execute')
for args, kwargs in common_args:
    dest = kwargs.get('dest')
    if dest in ['depends_on', 'depends_on_errors', 'label']: # not relevant to rest publish
        continue
    queue_attach_parser.add_argument(*args, **kwargs)

queue_list_parser = queue_parsers.add_parser('list', help='List queues')
queue_create_parser = queue_parsers.add_parser('create', help='Create queue')
queue_create_parser.add_argument('name', help='Name of new queue')
queue_delete_parser = queue_parsers.add_parser('delete', help='Delete queue')
queue_delete_parser.add_argument('name', help='Name of queue to delete')
"""

""" Cron Parser """
cron_parser = picloud_subparsers.add_parser('cron', description="Module for managing PiCloud Crons, periodically invoked functions on PiCloud", 
                                             help="Manage periodically invoked functions")
cron_parsers = cron_parser.add_subparsers(title='commands', dest='_command', help='command help')
cron_register_parser = cron_parsers.add_parser('register', help='Register a function which will be periodically invoked on PiCloud according to a cron schedule')
cron_register_parser.add_argument('label', help='Label to assign the registered cron')
cron_register_parser.add_argument('schedule', metavar='CRONTAB', help='Crontab schedule. See documentation for format details')
cron_register_parser.add_argument('command', nargs=argparse.PARSER, help='Shell command to execute')
for args, kwargs in common_args:
    dest = kwargs.get('dest')
    if dest in ['depends_on', 'depends_on_errors', 'label']: # not relevant to rest publish
        continue
    cron_register_parser.add_argument(*args, **kwargs)

cron_list_parser = cron_parsers.add_parser('list', help='List crons registered on PiCloud')
cron_info_parser = cron_parsers.add_parser('info', help='Retrieve information about a registered cron')
cron_info_parser.add_argument('label', help='Label of registered cron to get info about')

cron_deregister_parser = cron_parsers.add_parser('deregister', help='Deregister (delete) the cron specified by label')
cron_deregister_parser.add_argument('label', help='Label of registered cron to deregister')

cron_manualrun_parser = cron_parsers.add_parser('run', help='Manually run the command stored with a cron, ignoring schedule')
cron_manualrun_parser.add_argument('label', help="Label of registered cron's function to manually invoke")

"""Wait for"""
wait_for_parser = picloud_subparsers.add_parser('wait-for', description="Module for waiting on various aspects of a job. See subcommands", 
                                             help="Wait on aspects of a job")
wait_for_parsers = wait_for_parser.add_subparsers(title='commands', dest='_command', help='command help')
port_parser = wait_for_parsers.add_parser('port', description='Wait for job to open port for listening. Return translated external IP address and port', 
                                              help="Wait for port to open; and translate it")
port_parser.add_argument('-t', '--timeout', metavar='SECONDS', default=None, type=float,
                           help="Error if port is not ready by this number of seconds.")
port_parser.add_argument('-p', '--protocol', default='tcp', type=str, choices=['tcp', 'udp'],
                           help="Protocol port is listening on. Default is tcp")
port_parser.add_argument('jid', type=int, help="job identifier") 
port_parser.add_argument('port', type=int, help="Port to translate")

status_parser = wait_for_parsers.add_parser('status', description='Wait for job to have a a status', 
                                              help="Wait for status")
status_parser.add_argument('-t', '--timeout', metavar='SECONDS', default=None, type=float,
                           help="Error if port is not ready by this number of seconds.")
status_parser.add_argument('jid', type=int, help="job identifier")
from ..wait_for import _status_transitions 
status_parser.add_argument('test_status', metavar='status', type=str, choices=_status_transitions.keys(),
                           help='status to wait for')
