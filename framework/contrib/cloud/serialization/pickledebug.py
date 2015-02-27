"""
A debugging pickle object
This provides an xml trace of a pickle

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

import os
import types
import sys

import functools
import pickle
from pickle import PicklingError
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from . import xmlhandlers
from .cloudpickle import CloudPickler
from .. import util
from ..util import islambda
from .. import cloudconfig as cc

import logging
cloudLog = logging.getLogger("Cloud.Transport")

class DebugPicklingError(pickle.PicklingError):
    def __init__(self, msg, xmltrace):        
        pickle.PicklingError.__init__(self,msg)
        self.xmltrace = xmltrace
        
    def __str__(self):        
        return 'Pickle trace:\n' + self.xmltrace + '\n' + self.args[0] 


class CloudDebugPickler(CloudPickler):    
    
    typeAttributeFunctions = {} #type based attribute function
    typeCustomChildPrinter = {} #custom type printers for children
    
    #other properties:
    
    showPrimitives = cc.logging_configurable('show_primitives',
                                                       default=False,
                                                       comment="Should primitives be logged?",hidden=True)     
    printingMinSize = cc.logging_configurable('printing_min_size',
                                                        default=0,
                                                        comment="Minimum size objects should be to be logged",hidden=True)    
    showTransmittedTypesComposition =  \
        cc.logging_configurable('show_transmitted_types_composition',
                                       default=False,
                                       comment="Should composition of transmitted types be logged?",hidden=True)
        
    transmitted_forced_imports = None
    
    min_size_to_save = 0 #do not hold objects < min_size_to_save  
      
    
    def __init__(self, file, protocol=None):        
        CloudPickler.__init__(self,file,protocol)
        self.file = file
        self.traceStack = [] #stack to track as objects are added
        self.objSizes = {} #sizes of objects after pickling including sub data 
        self.objChildren = {} #objects pickled under this ID
        self.lastobj = None   
        self.aborting = False                            
    
    def dump(self, obj):
        
        try:
            CloudPickler.dump(self,obj)
        except PicklingError, p:            
            cloudLog.exception("pickling failed on %s of type %s." 
                           % (self.lastobj, type(self.lastobj)))
            strio = StringIO()                           
            
            xmlf = xmlhandlers.XmlWriter(strio,header=False)
            xmlmid = xmlhandlers.XmlStackWriter(xmlf)
            #xmlmid = xmlf
            
            self.aborting = True
            try: #intentionally crash at partial xml            
                self.dump_obj(xmlmid, obj, self.print_top_level, None, {}, abortObject = self.lastobj)                        
            except DebugPicklingError, e:
                if e.args[0] != 'fake exit':
                    raise     
            except RuntimeError, e: # dump_obj will crash w/ infinite recursion if regular dump did
                if 'recursion' not in e.args[0]:
                    raise
                                       
            self.aborting = False                
            xmlmid.flush()
            #raise
            raise DebugPicklingError(p.args[0],strio.getvalue())
        else:
            self.lastobj = None            
        
    
    def save_function_tuple(self, func, forced_imports):
        CloudPickler.save_function_tuple(self, func, forced_imports)
        if forced_imports:
            self.transmitted_forced_imports = forced_imports
    
    def save(self, obj):        
        #ignored types:        
        t = type(obj)
        
        #print 'saving...', len(self.traceStack), obj, type(obj)
        
        if id(obj) in self.memo: #deal with this better!!!            
            CloudPickler.save(self,obj)
            return
        
        self.lastobj = obj
        #block Numeric and boolean objects
        self.traceStack.append(id(obj))
        startpos = self.file.tell()
        
        #Do save:
        excp = None        
        try:            
            CloudPickler.save(self,obj)
        except PicklingError, p:               
            excp = p
            if id(obj) not in self.memo:  #forcibly memoize objects for debug printing..
                self.memoize(obj)

        endpos = self.file.tell()
        self.traceStack.pop()
        
        size = endpos - startpos
        ido = id(obj)
        if excp or size >= self.min_size_to_save:  #save if exception (to print debug) or larger than min_size
            self.objSizes[ido] = size
            if self.traceStack:        
                idp = self.traceStack[-1]
                if idp not in self.objChildren:
                    self.objChildren[idp] = []
                self.objChildren[idp].append(ido)
            if excp:
                raise excp
    
    
    """
    Reporting...
    Returns if an element was pushed
    """
    
    """XML Element Printers"""
    def print_dependency(self, xmlf, printerdata, obj, attrs):
        is_memo = attrs.get('is_memoized')         
        if not is_memo:  #hide memoized deps          
            attrs['objId'] = str(id(obj))              
            xmlf.push("Dependency",attrs)
            return True
        return False
        
    
    def print_class_attribute(self, xmlf, printerdata, obj, attrs):
        attrs.insertAt(0,'name',printerdata)
        xmlf.push('Attribute',attrs)
        return True

    def print_nontransmitted_attribute(self, xmlf, printerdata, obj, attrs):
        attrs.insertAt(0,'name',printerdata)
        xmlf.push('NonTransmittedAttribute',attrs)
        return True

    
    def print_collection(self, xmlf, printerdata, obj, attrs):
        """tuples and lists"""
        attrs.insertAt(0,'entryNum',str(printerdata))
        xmlf.push('Element',attrs)
        return True
    
    def print_dictionary_element(self, xmlf, printerdata, obj, attrs):
        """Dictionary"""
        attrs.insertAt(0,'key',printerdata[0])
        attrs.insertAt(1,'keyObjId',str(printerdata[1]))
        xmlf.push('Value',attrs)
        return True
                       

    def print_top_level(self, xmlf, printerdata, obj, attrs):
        xmlf.push("PickledObject",attrs)
        return True

    
    """
    Below methods define handlers to set control xml attributes for different types
    Each handler receives an the object to dump and XML element attributes to minipulate
    It returns childAttributes, allowDeps, childPrinter
    ChildAttributes is a dictionary of key/value pairs. the key is passed into the childPrinter and the value
        is the actual child
    If allowDeps is true, any dependencies this object has (outside of the childAttributes) are printed
    childPrinter is one of the before-defined xml printers that should be run
    
    childAttributes, allowDeps, childPrinter = attrFunction(self,obj, attrs)
    """
    
    def write_function(self, obj, attrs):        
        #attrs['type'] = "functionReference"         
        
        clist = self.objChildren.get(id(obj))
        
        if islambda(obj):            
            attrs['isLambda'] = str(True)
        else:
            attrs['funcName'] = str(obj.func_code.co_name)
        attrs['module'] = str(obj.__module__)            
        attrs['fileName'] = obj.func_code.co_filename
        attrs['firstLineNumber'] = str(obj.func_code.co_firstlineno)  
        if obj.func_code.co_freevars:
            attrs['closureVariables'] = pretty_print_str_iterator(obj.func_code.co_freevars) 

        #serialized function information:
        if clist:
            if len(clist) == 2:  #save global with attribute likely               
                m = self.memo.get(clist[1])[1]
                if isinstance(m, dict):
                    return m, False, self.print_class_attribute
            elif len(clist) > 2:                 
                attrs['TransmittedFunctionCode'] = str(True)
                #search for attributes. Expected ordering
                #code, globals, defaults closure, dict.  all but code can be missing
                #CodeType, dict, list, tuple/None, dict expected 
                i = 0
                clist = map(lambda x: self.memo.get(x,(None,None))[1],clist)
                max = len(clist)
                codeobj = None
                while i < max:                    
                    tst = clist[i]                                   
                    if type(tst) is types.TupleType and len(tst) == 2 and type(tst[0]) is types.CodeType:
                        codeobj = tst[1]
                        i+=1
                        break
                    i+=1
                if i == max: #extremely rare if even possible                    
                    return {}, True, self.print_class_attribute
                outdict = {}
                outdict['code'] = codeobj
                
                found = 0
                while i < max:
                    tst = clist[i]
                    typtst = type(tst)
                    if found <= 0 and typtst is types.DictType: #globals?
                        if i == max -1 and not obj.func_code.co_names: #might be attributes??
                            return {}, True, self.print_class_attribute #just list dependencies
                        #could still be attributes, but highly unlikely
                        outdict['globals'] = tst
                        found = 1
                    elif found <= 1 and typtst is types.TupleType: #defaults!
                        outdict['defaults'] = tst
                        found = 2
                    elif found <= 2 and typtst is types.ListType: #closure!
                        outdict['closure'] = tst
                        found = 3
                    elif found <= 3 and typtst is types.DictType:
                        outdict['function_attributes'] = tst
                        found = 4
                        break
                    i+=1
                
                return outdict, False, self.print_class_attribute
            
        return {}, True, self.print_class_attribute

    typeAttributeFunctions[types.FunctionType] = write_function
        
    def write_method(self, obj, attrs):
        """A method is a function bound to a class"""
        return {'function':obj.im_func,
                'self':obj.im_self,
                'class': obj.im_class},True,self.print_class_attribute
    typeAttributeFunctions[types.MethodType] = write_method            
       

    def write_partial(self, obj, attrs):
        return {'function': obj.func,
                'args': obj.args,
                'kwargs': obj.keywords
                },False,self.print_class_attribute

    typeAttributeFunctions[functools.partial] = write_partial

       
    def write_file(self, obj, attrs):
        """Transmitted file object"""
        attrs['filename'] = os.path.abspath(obj.name)
        return {},False,None
    typeAttributeFunctions[types.FileType] = write_file            
    
    
    def write_type_descriptor(self, obj, attrs):
        clist = self.objChildren.get(id(obj))
        if clist:
            attrs['transmittedDescriptor'] = str(True)   
            
        repr = str(obj)
        if repr[0:5] == '<type':
            attrs['type'] = 'typeDescriptor'
            attrs['typeDescribed'] = str(obj)[7:-2]
            #attrs['module'] = obj.__module__
            return {}, self.showTransmittedTypesComposition or self.aborting, None #Block out info?
        elif repr[0:6] == '<class':
            attrs['type'] = 'classDescriptorNew'
            attrs['classDescribed'] = str(obj)[8:-2]
            #attrs['module'] = obj.__module__
            return {}, self.showTransmittedTypesComposition or self.aborting, None #I'm not sure if this is how to do this?
        else: #???            
            #print 'wtf??', repr
            return {}, True, None    
    typeAttributeFunctions[types.TypeType] = write_type_descriptor
    
    def write_class_descriptor(self, obj, attrs):
        attrs['type'] = 'classDescriptorOld'
        attrs['classDesribed'] = str(obj)    
        #return obj.__dict__, True, self.print_class_attribute
        
        #check for type transmision:
        clist = self.objChildren.get(id(obj))
        if clist:
            attrs['transmittedDescriptor'] = str(True)        
        return {}, self.showTransmittedTypesComposition or self.aborting, None    
    typeAttributeFunctions[types.ClassType] = write_class_descriptor
    
    def write_raw_code(self, obj, attrs):
        #Suppress code blocks
        #attrs['type'] = 'pythonByteCode'
        return {}, False, None
    typeAttributeFunctions[types.CodeType] = write_raw_code

    def write_string(self, obj, attrs):
        if not self.showPrimitives and len(obj) < 256:
            attrs['hide'] = True
        if len(obj) < 1024:        
            attrs['contents'] = obj
        else:
            attrs['contents'] = obj[:1021] + '...'
        return {}, False, None
    typeAttributeFunctions[str] = write_string
    typeAttributeFunctions[unicode] = write_string
    
    def write_numeric(self, obj, attrs):    
        if attrs.has_key('implicitly_pickled'): #int's are generally not memoized - don't report it though
            del attrs['implicitly_pickled']
        if not self.showPrimitives:
            attrs['hide'] = True                
        attrs['contents'] = str(obj)                    
        return {}, False, None
    typeAttributeFunctions[int] = write_numeric
    typeAttributeFunctions[float] = write_numeric
    
    def write_complex(self, obj, attrs):    
        if not self.showPrimitives:
            attrs['hide'] = True                
        attrs['contents'] = str(obj.real) + '+' + str(obj.imag) + 'j'                    
        return {}, False, None    
    typeAttributeFunctions[complex] = write_complex

    
    def write_module(self, obj, attrs):
        attrs['module'] = obj.__name__
        return {}, False, None
    typeAttributeFunctions[types.ModuleType] = write_module
    
    def get_type_name(self, typ):
        if typ.__module__ != '__builtin__':
            return typ.__module__ + '.' + typ.__name__
        else:
            return typ.__name__
    
    def determine_collection_type(self, obj):
        """Determine the type of a sequence"""
        """Returns TYPE or MIXED"""        
        if not obj:
            return 'Empty'
        ctype = False
        for o in obj:
            if ctype is False:
                ctype = type(o)                
            elif type(o) != ctype:
                return 'Mixed'
        return self.get_type_name(ctype)
    
    def attr_collection(self, obj, attrs):
        attrs['numElements'] = str(len(obj)) 
        if obj:
            attrs['containedType'] = self.determine_collection_type(obj)
        i = 0
        outAttr = {}
        for c in obj:
            outAttr[i] = c
            i+=1
        return outAttr

    
    def write_sequence(self, obj, attrs):
        return self.attr_collection(obj,attrs),True,self.print_collection    
    typeAttributeFunctions[tuple] = write_sequence
    typeAttributeFunctions[list] = write_sequence


    def write_set(self, obj, attrs):
        return self.attr_collection(obj,attrs),False,self.print_collection   
    typeAttributeFunctions[set] = write_set
    typeAttributeFunctions[frozenset] = write_set
    
    def write_dictionary(self, obj, attrs):
        attrs['numElements'] = str(len(obj)) 
        if obj:
            attrs['keyType'] = self.determine_collection_type(obj.keys())
            attrs['valueType'] = self.determine_collection_type(obj.values())
        outAttr = {}
        for key, value in obj.items():
            typ = type(key)
            if typ == int or typ==float:
                attrname = str(key), id(key)
            elif typ == str:
                if len(key) > 128:
                    attrname = key[:64] + '...' + key[-64:], id(key)
                else:
                    attrname = key, id(key)
            else:
                attrname = 'object', id(key)
            outAttr[attrname] = value            
        return outAttr,True,self.print_dictionary_element    
    typeAttributeFunctions[dict] = write_dictionary
    
    def write_image(self, obj, attrs):
        """PIL Images"""
        attrs['type'] = 'PIL Image'
        attrs['resolution'] = str(obj.size)
        attrs['mode'] = str(obj.mode)
        if not obj.im and obj.fp and 'r' in obj.fp.mode and obj.fp.name \
            and not obj.fp.closed and (not hasattr(obj, 'isatty') or not obj.isatty()):
            attrs['transmitted_raw_data'] = str(False)
            return {'file': obj.fp}, False, self.print_class_attribute
        else:
            attrs['transmitted_raw_data'] = str(True)
            return {}, False, None
            
        
    
    def write_instance(self, obj, attrs):
        """For an old style class"""
        
        #hook for images:
        if hasattr(obj,'im') and hasattr(obj,'palette') and 'Image' in obj.__module__:
            return self.write_image(obj,attrs)
        
        #just handle state:
        try:
            getstate = obj.__getstate__
        except AttributeError:
            stuff = obj.__dict__
        else:
            stuff = getstate()
            
        if type(stuff) is not dict:
            stuff = {}
        return stuff, True, self.print_class_attribute
    typeAttributeFunctions[types.InstanceType] = write_instance
            
                
    def write_user_class(self, obj, attrs):
        """For a user defined new-style class - default"""
        if not issubclass(type(obj), types.TypeType) and hasattr(obj, '__module__') and hasattr(obj, '__reduce_ex__'):
            """New-style user class"""
            redf= obj.__reduce_ex__(2)
            if len(redf) >= 2 and getattr(redf[0],'__name__',"") == "__newobj__":
                if len(redf) >= 3:
                    attr = redf[2]
                if type(attr) is not dict:
                    attr = {}
                return attr, True, self.print_class_attribute
        #default behavior..
        if hasattr(obj,'__module__'):
            attrs['module'] = str(obj.__module__)            
        return {}, True, None
        
    def dump_obj(self, xmlf, obj, printer, printerdata, localMemo, abortObject = None):
        """Print an object
        xmlf is an XmlWriter
        obj is the object to print
        printer is a method describing how to print this object
        printerdata is data used by the printer object
        if memoized_ref is true, a reference is printed
        """       
        #print obj, abortObject, abortObject is obj
        typ = type(obj)
        attrs = util.OrderedDict()
        ido = id(obj)                    
        sz = self.objSizes.get(ido)
        if sz == None: #todo... return?
            #print 'null sz'
            sz = 0 
        if sz < self.printingMinSize and (not abortObject or abortObject is not obj): #hide small objects
            return

        memo_entry = self.memo.get(ido)        
        
        attrs['type'] = self.get_type_name(type(obj))
        
        memoized_ref = ido in localMemo
        if sz == 0: #non-physical object
            attrs['not_pickled'] = str(True)
        elif obj is None:
            pass
        elif not memo_entry:  #implicit pickling -- reduced to another object
            attrs['implicitly_pickled'] = str(True)
            if memoized_ref:
                attrs['is_memoized'] = str(True)
            else:
                attrs['size'] =  str(sz) 
                localMemo[ido] = True        
        elif memoized_ref:
            attrs['is_memoized'] = str(True)
            attrs['memo_id'] = str(localMemo[ido])  #memo_len identiy                
        else:      #new object            
            attrs['size'] =  str(sz)     
            if memo_entry is not None:
                attrs['memo_id'] = str(memo_entry[0])  #memo_len identiy
                localMemo[ido] = memo_entry[0]            
                                            
        attrFunction = self.typeAttributeFunctions.get(typ)   
        #print 'typ', typ, attrFunction     
        if attrFunction:
            childAttributes, allowDeps, childPrinter = attrFunction(self,obj, attrs)
        else:                       
            childAttributes, allowDeps, childPrinter = self.write_user_class(obj,attrs)
            #print 'user', childAttributes, allowDeps, childPrinter      
        
        """
        if isinstance(childAttributes, tuple):
                    print childAttributes, attrFunction.__name__, typ
        """ 
        
        if abortObject and abortObject is obj:
            attrs.insertAt(0,'failed_to_pickle',str(True))            
        elif attrs.get('hide'):
            return
             
        needPop = printer(xmlf, printerdata, obj, attrs)  #push xml element
        
        if abortObject and abortObject is obj:
            #print 'AB',  abortObject
            raise DebugPicklingError('fake exit',None)
        
        if not memoized_ref:
            customPrinter = self.typeCustomChildPrinter.get(typ)
            if customPrinter is not None: 
                #custom Printers are responsible for all child serialization
                customPrinter(self,xmlf,obj)
            else:                          
                #function below??:            
                clearedDeps = [] #list of dependency ids already printed as attributes        
                clist = self.objChildren.get(ido) #list of dependencies of this object 
                #print '\nclist', clist
                #attributes:      
                for key, value in childAttributes.items():
                    idc = id(value)
                    self.dump_obj(xmlf,value,childPrinter,key,localMemo,abortObject)
                    clearedDeps.append(idc)
                if allowDeps:  #other dependencies..
                    if clist:
                        for idc in clist:
                            if idc in clearedDeps:
                                continue                            
                            child_memo_entry = self.memo.get(idc)                            
                            if child_memo_entry is not None:
                                cobj = child_memo_entry[1]
                                #validate tuples/lists/dicts:
                                if type(cobj) == tuple or type(cobj) == list:
                                    if all([id(subobj) in clearedDeps for subobj in cobj]):
                                        continue
                                elif type(cobj) == dict:
                                    #verify if this is an attribute list
                                    if all([type(key) == str for key in cobj.keys()]):
                                        if all([id(subobj) in clearedDeps for subobj in cobj.values()]):
                                            continue                                    
                                self.dump_obj(xmlf,cobj,self.print_dependency,None,localMemo,abortObject)
        if needPop:
            if printer == self.print_top_level:
                #always show transmitted_forced_imports at top level
                if self.transmitted_forced_imports:
                    xmlf.push("forced_imports")
                    for mod in self.transmitted_forced_imports:
                        xmlf.push('item',{'module':mod.__name__})
                        xmlf.pop()
            xmlf.pop() #pop printed entry
    
    def write_report(self, topobj, outfile, abortObject = None, hideHeader=False):
        """Write an xml report about the pickling of topobj to outfile"""           
        if not abortObject:  #if there is no explicit abortObject use lastobj (if any)
            abortObject = self.lastobj                 
        xmlf = xmlhandlers.XmlWriter(outfile,header=not hideHeader)
        if not hideHeader:
            xmlf.comment('Properties',{'showPrimitives':str(self.showPrimitives),
                                   'minimumObjectSizePrinted':str(self.printingMinSize),
                                   'typeCompositionShown':str(self.showTransmittedTypesComposition)})                                   
        self.aborting = True
        try:
            self.dump_obj(xmlf, topobj, self.print_top_level, None, {}, abortObject = abortObject)
        except DebugPicklingError: #ignore aborts
            pass
        self.aborting = False
        
def pretty_print_str_iterator(str_iterator):
    k = '['
    start = False
    for str in str_iterator:
        if start:
            k+=','
        k+=str
        start = True
        
    k+=']'
    return k

#shorthands for legacy support

def dump(obj, file, protocol=2):
    CloudDebugPickler(file, protocol).dump(obj)

def dumps(obj, protocol=2, debugprint = False):    
    file = StringIO()        
    cp = CloudDebugPickler(file, protocol)
    cp.dump(obj)
        
    #if type(obj) is not dict and type(obj) is not list and type(obj) is not tuple:
    if debugprint:     
        print 'reporting', type(obj), obj
        cp.write_report(obj,sys.stdout) 
    return file.getvalue()

