"""
Various utility functions

Copyright (c) 2009 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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
import types
import cPickle
import inspect
import datetime
import os

from functools import partial
from warnings import warn

def islambda(func):
    return getattr(func,'func_name') == '<lambda>'

def funcname(func):
    """Return name of a callable (function, class, partial, etc.)"""
    module = ""
    if hasattr(func,'__module__'):
        module = (func.__module__ if func.__module__ else '__main__')
    """Return a human readable name associated with a function"""
    if inspect.ismethod(func):
        nme = '.'.join([module,func.im_class.__name__,func.__name__]) 
    elif inspect.isfunction(func):
        nme =  '.'.join([module,func.__name__])
    elif inspect.isbuiltin(func):
        return  '.'.join([module,func.__name__])
    elif isinstance(func,partial):
        return 'partial_of_' + funcname(func.func)    
    elif inspect.isclass(func):
        nme = '.'.join([module,func.__name__])
        if hasattr(func, '__init__') and inspect.ismethod(func.__init__):            
            func = func.__init__
        else:
            return nme #can't extract more info for classes
        
    else:
        nme = 'type %s' % type(func)
        if hasattr(func, '__name__'):
            nme = '%s of %s' % (func.__name__, type(func))
        return nme
    nme +=  ' at ' + ':'.join([func.func_code.co_filename,str(func.func_code.co_firstlineno)])
    return nme 
    
def min_args(func):
    """Return minimum (required) number args this function has"""
    if inspect.isfunction(func):
        op_args = len(func.func_defaults) if func.func_defaults else 0 
        return func.func_code.co_argcount - op_args
    elif inspect.ismethod(func):
        return min_args(func.im_func) - 1
    elif inspect.isclass(func):
        if hasattr(func, '__init__'): #check class constructor
            return min_args(func.__init__)
        else:
            return 0
            
    raise TypeError('cannot deal with type: %s' % type(func))

def max_args(func):
    """Return maximum (required + default) number of arguments callable can take"""
    if inspect.isfunction(func):
        return func.func_code.co_argcount
    elif inspect.ismethod(func):
        return max_args(func.im_func) - 1
    elif inspect.isclass(func) and hasattr(func, '__init__'): #check class constructor
        if hasattr(func, '__init__'): #check class constructor
            return max_args(func.__init__)
        else:
            return 0
    raise TypeError('cannot deal with type: %s' % type(func))


def getargspec(func):
    """Returns an argspec or None if it can't be resolved
    Our argspec is similar to inspect's except the name & if it is a method is appended as the first argument
    Returns (name, is_method, args, *args, **kwargs, defaults)
    """
    
    try:
        argspec = inspect.getargspec(func)
    except TypeError:
        return None

    out_list = [func.__name__, int(inspect.ismethod(func))]
    out_list.extend(argspec)
    return out_list
    
def validate_func_arguments(func, test_args, test_kwargs):
    """First pass validation to see if args/kwargs are compatible with the argspec
    Probably doesn't catch everything that will error
    Known to miss:
        Validate that anonymous tuple params receive tuples
    
    This is only valid for python 2.x
    
    Returns true if validation passed; false if validation not supported
        Exception raised if validation fails
    """
    try:
        argspec = inspect.getargspec(func)
    except TypeError: #we can't check non-functions
        return False
        
    return validate_func_arguments_from_spec( (func.__name__, int(inspect.ismethod(func))) + argspec, 
                                              test_args, 
                                              test_kwargs.keys())
    
    
def validate_func_arguments_from_spec(argspec, test_args, test_kwargs_keys):
    
    name, is_method, args, varargs, varkw, defaults = argspec
    
    if defaults == None:
        defaults = []
    else:
        defaults = list(defaults)
    
    if is_method: #ignore self/cls
        args = args[1:]
                
    
    name += '()' #conform to python error reporting
    test_args_len = len(test_args)   
    
    #kwd exist?
    if not varkw:
        for kw in test_kwargs_keys:
            if kw not in args:
                raise TypeError("%s got an unexpected keyword argument '%s'" % (name, kw))
        
    #kwd not already bound by passed arg?
    kwd_bound = args[test_args_len:] #These must all be default or bound to kwds
    if not varkw:
        for kw in test_kwargs_keys:
            if kw not in kwd_bound:
                raise TypeError("%s got multiple values for keyword argument '%s'" % (name, kw))
    
    #verify argument count
    firstdefault = len(args) - len(defaults)
    nondefargs = args[:firstdefault]
    
    defaults_injected = 0
    for kw in test_kwargs_keys:
        if kw in nondefargs:
            defaults.append(None)  #pretend another default is there for counting
            defaults_injected += 1

    min = len(args) - len(defaults)
    max = len(args)
    
    #correct for default injection
    min+=defaults_injected
    max+=defaults_injected
    test_args_len += defaults_injected
    
    if varargs:
        max = sys.maxint
    if min < 0:
        min = 0
    
    if test_args_len < min or max < test_args_len:
        err_msg = '%s takes %s arguments (%d given)'
        
        if min == max:
            arg_c_msg = 'exactly %s' % min        
        elif test_args_len < min:
            arg_c_msg = 'at least %s' % min
        else:
            arg_c_msg = 'at most %s' % max
        
        raise TypeError(err_msg % (name, arg_c_msg, test_args_len))
    
    return True

def fix_time_element(dct, key):
    """Fix time elements in dictionaries coming off the wire"""
    item = dct.get(key)
    if item == 'None': #returned by web instead of a NoneType None
        item = None
        dct[key] = item
    if item:
        dct[key] = datetime.datetime.strptime(item,'%Y-%m-%d %H:%M:%S')
    return dct

def fix_sudo_path(path):
    """Correct permissions on path if using sudo from another user and keeping old users home directory"""
    
    if os.name != 'posix': 
        return
    
    sudo_uid = os.environ.get('SUDO_UID')
    sudo_user = os.environ.get('SUDO_USER')
    
    if sudo_uid != None and sudo_user:
        sudo_uid = int(sudo_uid)
        home = os.environ.get('HOME')
        sudo_user_home = os.path.expanduser('~' + sudo_user)
        
        # important: Only make modifications if user's home was not changed with sudo (e.g. sudo -H)
        if home == sudo_user_home:
            sudo_gid = os.environ.get('SUDO_GID')
            sudo_gid = int(sudo_gid) if sudo_gid else -1
            try:
                os.chown(path, sudo_uid, sudo_gid)
            except Exception, e:
                warn('PiCloud cannot fix SUDO Paths. Error is %s:%s' % (type(e), str(e)))
                
   
"""Ordered Dictionary"""
import UserDict
class OrderedDict(UserDict.DictMixin):
    
    def __init__(self, it = None):
        self._keys = []
        self._data = {}
        if it:
            for k,v in it:
                self.__setitem__(k,v)
        
        
    def __setitem__(self, key, value):
        if key not in self._data:
            self._keys.append(key)
        self._data[key] = value
    
    def insertAt(self, loc, key, value):
        if key in self._data:
            del self._data[self._data.index(key)]
        self._keys.insert(loc, key)
        self._data[key] = value
        
    def __getitem__(self, key):
        return self._data[key]
    
    
    def __delitem__(self, key):
        del self._data[key]
        self._keys.remove(key)
        
        
    def keys(self):
        return list(self._keys)    
    
    def copy(self):
        copyDict = OrderedDict()
        copyDict._data = self._data.copy()
        copyDict._keys = self._keys[:]
        return copyDict
    
"""Python 2.5 support"""
from itertools import izip, chain, repeat
if sys.version_info[:2] < (2,6):
    def izip_longest(*args):
        def sentinel(counter = ([None]*(len(args)-1)).pop):
            yield counter()         # yields the fillvalue, or raises IndexError
        fillers = repeat(None)
        iters = [chain(it, sentinel(), fillers) for it in args]
        try:
            for tup in izip(*iters):
                yield tup
        except IndexError:
            pass

if __name__ == '__main__':
    """Validate the validate_func_arguments function"""
    def foo0():
        pass
    
    def foo1(a):
        pass
    
    def foo2(a, b=2):
        pass

    def foo21(a, b):
        pass
    
    def foo3(a, (x,y), b):
        """lovely anonymous function"""
        pass
    
    def consist(func, *args, **kwargs):
        typerror = None
        try:
            func(*args, **kwargs)
        except TypeError, e:
            typerror = e
        
        print '%s %s %s' % (func, args, kwargs)
        try:
            validate_func_arguments(func, args, kwargs)
        except TypeError, e:
            if not typerror:
                print 'unexpected typerror! %s' % str(e)
                raise
            else:
                print '%s == %s' % (typerror, str(e))
        else:
            if typerror:
                print 'missed error! %s' % typerror
                raise
            else:
                print 'no error!'
                
    consist(foo0)
    consist(foo0, 2)
    consist(foo0, k=2)
    consist(foo0, 3, k=4)
    
    consist(foo1)
    consist(foo1, b=2)
    consist(foo1, a=2)
    consist(foo1, 2)
    consist(foo1, 3)
    consist(foo1, 3, a=2)
    consist(foo1, 3, b=2)
    
    consist(foo2)
    consist(foo2, b=2)
    consist(foo2, b=2, c=3)
    consist(foo2, a=2)
    consist(foo2, a=2, b=2)
    consist(foo2, a=2, b=2, c=3)
    consist(foo2, 2, a=10)
    consist(foo2, 3)
    consist(foo2, 3, 4)
    consist(foo2, 3, 4, 7)
    consist(foo2, 3, b=2)
    consist(foo2, 3, a=10, b=2)
    consist(foo2, 3, b=2, c=2)
    consist(foo2, 3, a=10, b=2, c=4)
    
    consist(foo21, 3, 4)
    consist(foo21, 3, b=4)
    consist(foo21, a=3, b=4)
    consist(foo21, b=4)
    consist(foo21, a=4)
    consist(foo21)
    consist(foo21, 4, 3, 5)
    
    consist(foo3, 2, (4,3), 9)
    consist(foo3, 2, (4,3), b=9)
    consist(foo3, 2, (4,3), a=9)
    consist(foo3, 2, (4,3), a=9, b=9)
    consist(foo3, 2,  a=9, b=9)
    consist(foo3, 2, (4,3))
    
    #we can't catch below..
    #consist(foo3, 2,  10, 12)
    
