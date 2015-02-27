from __future__ import with_statement

'''
Provides for storage and retrieval of PiCloud credentials

Current credentials include:
- cloudauth: key/secretkey
- ssh private keys (environments/volumes) 
'''
import distutils
import os

from ConfigParser import RawConfigParser
from .. import cloudconfig as cc
import logging
cloudLog = logging.getLogger('Cloud.credentials')


credentials_dir = os.path.expanduser(os.path.join(cc.baselocation,'credentials'))

"""general"""
key_cache = {}  # dictionary mapping key to all authentication information

def save_keydef(key_def, api_key=None):
    """Save key definition to necessary files. Overwrite existing credential
        If *api_key* not None, verify it matches key_def
    """
    key_def['api_key'] = int(key_def['api_key'])
    if not api_key: 
        api_key = key_def['api_key']
    else:    
        assert (key_def['api_key'] == int(api_key))    
    key_cache[api_key] = key_def
    write_cloudauth(key_def) #flush authorization
    write_sshkey(key_def)    #flush ssh key
    
def download_key_by_key(api_key, api_secretkey):
    """Download and cache key""" 
    api_key = int(api_key)       
    from ..account import get_key_by_key
    key_def = get_key_by_key(api_key, api_secretkey)
    cloudLog.debug('Saving key for api_key %s' % api_key)
    save_keydef(key_def, api_key)
    return key_def

def download_key_by_login(api_key, username, password):
    """Download and cache key by using PiCloud login information""" 
    api_key = int(api_key)       
    from ..account import get_key
    key_def = get_key(username, password, api_key)
    save_keydef(key_def, api_key)
    return key_def

def verify_key(api_key):
    """Return true if we have valid sshkey and cloudauth for this key.
    False if any information is missing"""
    key_def = key_cache.get(api_key, {})
    if 'api_secretkey' not in key_def:    
        if not resolve_secretkey(api_key):
            cloudLog.debug('verify_key failed: could not find secretkey for %s', api_key)
            return False
    if not 'private_key' in key_def:
        res = verify_sshkey(api_key)
        if not res:
            cloudLog.debug('verify_key failed: could not find sshkey for %s', api_key)
        return res

def get_credentials_path(api_key):
    """Resolve directory where credentials are stored for a given api_key
    Create directory if it does not exist"""
    path = os.path.join(credentials_dir, str(api_key))
    try:
        distutils.dir_util.mkpath(path)
    except distutils.errors.DistutilsFileError:
        cloudLog.exception('Could not generate credentials path %s' % path)
    return path

""" Api keys"""
#constants:
api_key_section = 'ApiKey'

def get_cloudauth_path(api_key):
    """Locate cloudauth path"""
    base_path = get_credentials_path(api_key)
    return os.path.join(base_path, 'cloudauth')

def read_cloudauth(api_key):
    """Load cloudauth for api_key"""
    path = get_cloudauth_path(api_key)
    if not os.path.exists(path):
        raise IOError('path %s not found' % path)
    config = RawConfigParser()
    config.read(path)
    
    key_def = key_cache.get(api_key, {})
    key = config.getint(api_key_section, 'key')
    if key != api_key:
        raise ValueError('Cloudauth Credentials do not match. Expected key %s, found key %s' % (api_key, key))
    key_def['api_key'] = key
    key_def['api_secretkey'] = config.get(api_key_section, 'secretkey')
    key_cache[int(api_key)] = key_def
    return key_def

def get_saved_secretkey(api_key):
    """Resolve the secret key for this api_key from the saved cloudauth credentials"""
    api_key = int(api_key)
    key_def = key_cache.get(api_key)
    if not key_def:
        key_def = read_cloudauth(api_key)
    return key_def['api_secretkey']

def write_cloudauth(key_def):
    """Write key/secret key information defined by key_def into cloudauth"""
    api_key = str(key_def['api_key'])
    api_secretkey = key_def['api_secretkey']    
    path = get_cloudauth_path(api_key)
        
    config = RawConfigParser()
    config.add_section(api_key_section)
    config.set(api_key_section, 'key', api_key)
    config.set(api_key_section, 'secretkey', api_secretkey)
    try:
        with open(path, 'wb') as configfile:
            config.write(configfile)
    except IOError, e:
        cloudLog.exception('Could not save cloudauth credentials to %s' % path)
    try:
        os.chmod(path, 0600)
    except:
        cloudLog.exception('Could not set permissions on %s' % path)
        

def resolve_secretkey(api_key):
    """Find secretkey for this api_key
    Return None if key cannot be found
    """    
    try:
        secretkey = get_saved_secretkey(api_key)
    except Exception, e:
        if not isinstance(e, IOError):
            cloudLog.exception('Unexpected error reading credentials for api_key %s' % api_key)
        return None
    else:
        return secretkey        


""" SSH private keys 
These private keys are used to connect to PiCloud
"""    

def get_sshkey_path(api_key):
    """Locate where SSH key is stored"""
    base_path = get_credentials_path(api_key)
    return os.path.join(base_path,'id_rsa')

def read_sshkey(api_key):
    """Read sshkey from file.
    Save to cache and return key_def. key will be in key_def['private_key']"""
    path = get_sshkey_path(api_key)    
    with open(path, 'rb') as f:
        private_key = f.read()
    key_def = key_cache.get(api_key, {})
    key_def['api_key'] = api_key
    key_def['private_key'] = private_key
    key_cache[int(api_key)] = key_def
    return key_def
    
def verify_sshkey(api_key):
    """Verify sshkey presence
    Todo: Actually validate key    
    """
    path = get_sshkey_path(api_key)
    if os.path.exists(path):
        try:
            os.chmod(path, 0600)
        except:
            cloudLog.exception('Could not set permissions on %s' % path)
        return True
    return False

def write_sshkey(key_def):
    """Save key_def['private_key'] to sshkey_path"""
    private_key = key_def['private_key']
    api_key = key_def['api_key']
    path = get_sshkey_path(api_key)    
    try:
        with open(path, 'wb') as f:
            f.write(private_key)
    except IOError, e:
        cloudLog.exception('Could not save ssh private key to %s' % path)
    else:
        try:
            os.chmod(path, 0600)
        except:
            cloudLog.exception('Could not set permissions on %s' % path)
        
    
def test(key, secretkey):
    key_has = verify_key(key)
    print 'have key already? %s' % key_has 
    if not key_has:
        print 'downloading'
        download_key_by_key(key, secretkey)
        key_has = verify_key(key)
        print 'have key now? %s' % key_has

    secretkey = resolve_secretkey(key)
    print 'your key is %s' % secretkey
        