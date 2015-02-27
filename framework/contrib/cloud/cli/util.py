import sys
import logging
import textwrap
from itertools import izip

try:
    import json
except ImportError:
    import simplejson as json
    
def safe_str(x):
    if isinstance(x, basestring):
        return x
    else:
        return str(x)    
    
def truncate(x):
    x = safe_str(x)
    x.strip()
    if '\n' in x:
        x, _ = x.split('\n', 2)
        x += ' ...'
    return x    

def safe_print(s):
    try:
        print s
    except UnicodeEncodeError:
        print s.encode(sys.stdout.encoding if sys.stdout.encoding else 'utf-8', errors='replace')

def safe_print_noline(s):
    try:
        print s,
    except UnicodeEncodeError:
        print s.encode(sys.stdout.encoding if sys.stdout.encoding else 'utf-8', errors='replace'),


class NullHandler(logging.Handler):
    """Does nothing with log messages. Included by default in the Python
    library for version 2.7+, but necessary for 2.6 compatibility"""
    
    def emit(self, record):
        pass

def list_of_dicts_printer(headers):
    """Prints a list of dictionaries as a table with an optional header"""
    
    def list_of_dicts_printer_helper(l, print_headers=True, kwargs={}):
        
        l = list(l)  # need to iterate twice through it
        spacing = {}
        for header in headers:
            max_column_width = len(header) if print_headers else 0
            for item in l:
                max_column_width = max(max_column_width, len(truncate(item.get(header, ''))))
            spacing[header] = max_column_width
        
        if print_headers:
            for header in headers:
                safe_print_noline( '%s%s'  % (header, ' ' * (spacing[header] - len(header) + 4)) )
            safe_print( '' )
        
        for item in l:
            for header in headers:
                safe_print_noline( '%s%s'  % (truncate(item.get(header, '')), ' ' * (spacing[header] - len(truncate(item.get(header, ''))) + 4)))
            print ''
    
    return list_of_dicts_printer_helper

def dict_printer(headers):
    """Prints a dictionary as a table with one row with an optional header"""
    
    def dict_printer_helper(d, print_headers=True, kwargs={}):
        list_of_dicts_printer(headers)([d], print_headers)
    
    return dict_printer_helper

def key_val_printer(key_header, value_headers):
    """Print a dictionary with keys mapping to value (or list of values).
    or dictionary of values. (if dictionary, value_headers should be superset of possible keys in value dictionary)    
    """
    
    headers = [key_header]
    
    if hasattr(value_headers, '__iter__'):
        headers.extend(value_headers)
        val_is_list = True
    else:
        headers.append(value_headers) 
        val_is_list = False
        
    def list_of_dicts_gen_singleval(d):
        """Where values of d are items"""
        return [{key_header : k, value_headers: v} for k, v in d.items()]
        
    def list_of_dicts_gen_list(d):
        """Where values of d are lists"""
        out_dcts = []
        for k, v in d.items():
            dct = dict(((vh, val) for vh, val in izip(value_headers, v)))
            dct[key_header] = k
            out_dcts.append(dct)
        return out_dcts              
    
    def list_of_dicts_gen_dict(d):
        """Where values of d are dictionaries"""
        out_dcts = []
        for k, v in d.items():
            dct = {key_header : k}
            dct.update(v)
            out_dcts.append(dct)
        return out_dcts              
        
    def helper(d, print_headers=True, kwargs={}):
        if val_is_list and isinstance(d.values()[0], dict):
            found_headers = set()
            for x in d.values():
                found_headers.update(x)
            return list_of_dicts_printer(found_headers)(list_of_dicts_gen_dict(d), print_headers)
                            
        lst_dct_gen = list_of_dicts_gen_list if val_is_list else list_of_dicts_gen_singleval
        return list_of_dicts_printer(headers)(lst_dct_gen(d), print_headers)
    
    return helper    
     

def list_printer(header):
    """Prints a list with one element per line with an optional header"""
    
    def list_printer_helper(l, print_headers=True, kwargs={}):
        
        if print_headers:
            safe_print(header)
        
        for item in l:
            safe_print(item)
        
        if getattr(l,'truncated', False):
            print >> sys.stderr, '...Results are truncated...'        
        
    return list_printer_helper

def volume_ls_printer(listings, print_headers=False, kwargs={}):
    """Prints the result of volume ls."""
    entries_printer = list_of_dicts_printer(['name', 'size', 'modified'])
    num_paths = len(listings)
    for i in range(num_paths):
        path, listing = listings[i]
        print path
        print 'total %s' % len(listing)
        entries_printer(listing, print_headers=False)
        if i < (num_paths - 1):
            print

    
            
def cloud_info_printer(info_results, print_headers, kwargs):
    """Dump contents"""
    
    base_ordering = ['status', 'exception', 'stdout', 'stderr', 'runtime', 
                     'created', 'finished', 'code_version', 'env', 'vol', 'exception', 'profile',
                
                ]
    info_ordering = kwargs.get('info_requested')
    if not info_ordering:
        info_ordering = base_ordering
    else:
        info_ordering = info_ordering.split(',')
        
    nl_info = ['stdout', 'stderr', 'exception', 'profile']
    
    started = False
    for jid, info_result in info_results.items():
        if started:
            print
        started = True
        
        # dyanamically update ordering as needed:
        for key in info_result:
            if key not in info_ordering:
                info_ordering.append(key)                
        
        print 'Info for jid %s' % jid                
        for info_key in info_ordering:
            result = info_result.get(info_key)
            if result:
                if info_key in nl_info:                
                    print '%s:' % info_key
                    safe_print(result)
                else:
                    safe_print('%s: %s' % (info_key, result))            

def bucket_info_printer(bucket_info_result, print_headers, kwargs):
    ordering = ['size', 'created', 'last-modified', 'md5sum', 'public', 'url']
    
    for key in bucket_info_result:
        if key not in ordering:
            ordering.append(key)                
                    
    for info_key in ordering:
        try:
            result = bucket_info_result[info_key]
        except KeyError:
            continue
        else:
            safe_print('%s: %s' % (info_key, result))            

def cloud_result_printer(results, print_headers, kwargs):
    started = False
    if len(results) == 1: # if just one result, print it directly (allows binary data to be written)
        val = results.values()[0]
        if isinstance(val, basestring):
            sys.stdout.write(val)
        else: # add newline if non-string
            print val 
    else:
        for jid, result in results.items():
            if started:
                print
            started = True
            print 'Result for jid %s:' % jid
            if isinstance(result, basestring):
                safe_print_noline(result)
            else: # add newline if non-string
                safe_print(result) 

    
def cloud_result_json_printer(results, print_headers, kwargs):
    """Values of results may already json encoded, hence the need for a special printer"""
    print '{',
    started = False 
    new_results = {}
    
    """First pass: Verify that all values are representable in json"""
    for jid, result in results.items():
        if not isinstance(result, basestring):
            result = json.dumps(result) # must be a primitive type at this stage
        elif not getattr(result, 'json_encoded', None): # functions.result marks results it encoded
            try:
                result = json.dumps(result)
            except (TypeError, UnicodeDecodeError):
                raise ValueError('result of jid %s cannot be represented in json. Please use default output format')
        new_results[jid] = result        
       
    for jid, result in new_results.items():
        if started:
            print ',',
        started = True
        safe_print_noline('"%s":' % jid)
        safe_print_noline('%s' % result)
            
    print '}'

def no_newline_printer(results, print_headers, kwargs):
    if results is None:
        pass
    else:
        print results,

from _abcoll import KeysView, ItemsView, ValuesView, MutableMapping

try:
    from thread import get_ident as _get_ident
except ImportError:
    from dummy_thread import get_ident as _get_ident
