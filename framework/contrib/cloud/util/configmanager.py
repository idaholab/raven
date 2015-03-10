"""
PiCloud configuration backend

Copyright (c) 2010 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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

class NoOptionError(Exception):
    """A requested option was not found."""

    def __init__(self, option):
        Exception.__init__(self, "No key %r" % option)
        self.option = option

extraInfo = {
             'Account': 'PiCloud account information. This is the only section that you need to worry about.',
             'Logging': 'Control what should be logged and where',
             'Transport': 'PiCloud information transfer',
             'Multiprocessing': 'Options that control running the cloud locally',
             'Simulation': 'Options for simulation mode that override Multiprocessing and Logging options'
        }


class ConfigManager(object):
        
    backend = None
    hiddenSets = []
    
    @staticmethod
    def getCommentStr(section, option):
        return option.lower()
    
    def __init__(self, defaults=None):
        self.sections = {}
        self.optioncomment = {}
    
    def read(self, fname):
        """Return True on successful read"""
        import os
        import sys
        dir = os.path.dirname(fname)
        conf = os.path.basename(fname)
        pyfile = os.path.splitext(conf)[0]                   
        addedEntry = False
                
        try:
            if dir not in sys.path:
                sys.path.append(dir)
                addedEntry = True
            if not os.path.exists(fname):
                try:
                    os.unlink("".join([dir, os.sep, pyfile, '.pyc'])) #force recompilation
                except OSError:
                    pass
                import types
                self.backend = types.ModuleType('cloudconf')
                return False #force rewrite
            else:                    
                try:
                    if pyfile in sys.modules:
                        self.backend = sys.modules[pyfile]
                    else:
                        self.backend = __import__(pyfile)
                except ImportError, e:
                    import types
                    sys.stderr.write('CLOUD ERROR: Malformed cloudconf.py:\n %s\nUsing default settings.\n' % str(e))
                    self.backend = types.ModuleType('cloudconf')            
        finally:    
            if addedEntry:
                sys.path.remove(dir)
        return True
    
    def get(self, section, option, comment = None):
        if not hasattr(self.backend, option):
            raise NoOptionError(option)
        value = getattr(self.backend, option)
        self.sections.setdefault(section, {})[option] = value
        if comment:
            self.optioncomment[self.getCommentStr(section, option)] = comment
        return value
    
    def hiddenset(self, *args): 
        """Defer set commands"""
        self.hiddenSets.append(args)
        
    def showHidden(self):
        """Do all deferred (hidden) sets -- not thread safe"""
        for hiddenSet in self.hiddenSets:
            self.set(*hiddenSet)
        self.hiddenSets = []
    
    def set(self, section, option, value, comment = None):        
        self.sections.setdefault(section, {})[option] = value    
        if comment:
            self.optioncomment[self.getCommentStr(section, option)] = comment
        #print 'setting backend %s to %s' % (option, value)
        setattr(self.backend,option,value)
            
    def write(self, fp):
        """Write configuration file with defaults
        Include any comments"""        
        #hack to ensure account comes first:
        sections = self.sections.keys()
        sections.sort()
        
        for section in sections:
            cmt = '"' * 3
            fp.write('%s\n%s\n' % (cmt, section))
            ei = extraInfo.get(section)
            if ei:
                fp.write('%s\n%s\n' % (ei, cmt))
            else:
                fp.write('%s\n' % cmt)
            started = False
            for (key, value) in self.sections[section].items():                
                if key != "__name__":                    
                    comment = self.optioncomment.get(self.getCommentStr(section, key))                  
                    if comment:
                        if started:
                            fp.write('\n')
                        for cel in comment.split('\n'):
                            fp.write('# %s\n' % cel.strip())
                    #print 'write %s=%s with type %s'% (key, repr(value), type(value))
                    fp.write("%s = %s\n" %
                             (key, repr(value).replace('\n', '\n\t')))
                    started = True
            fp.write("\n\n")

class ConfigSettings(object):
    """This object provides the ability to programmatically edit the cloud configuration (found in cloudconf.py).   
    ``commit()`` must be called to update the cloud module with new settings - and restart all active clouds    
    """
    
    
    @staticmethod
    def _loader(path,prefix, do_reload):        
        """Bind """               
        files = os.listdir(path)
        delayed = []
        for f in files:
            if f.endswith('.py'):
                endname = f[:-3]
                if endname == 'cloudconfig' or endname == 'configmanager' or endname == 'setup' or endname == 'writeconfig' or endname == 'cli':                    
                    continue
                if endname == '__init__':
                    delayed.append(prefix[:-1])  #do not load __init__ until submodules reloaded
                    continue
                elif endname == 'mp':
                    modname = prefix + endname
                    delayed.append(modname)
                else:                
                    modname = prefix + endname
                #print modname #LOG ME   
                if do_reload:
                    if modname in sys.modules:
                        try:
                            reload(sys.modules[modname])
                        except ImportError:
                            pass
                else:
                    try:
                        __import__(modname)
                    except ImportError:
                        pass                    
            elif os.path.isdir(path + f):
                newpath = path + f + os.sep
                ConfigSettings._loader(newpath,prefix + f + '.',do_reload)
        if delayed:
            if '__init__' in delayed: #must come last
                delayed.remove('__init__')
                delayed.append('__init__')
                
            for delay_mod in delayed:
                if do_reload:
                    if delay_mod in sys.modules:
                        try:
                            reload(sys.modules[delay_mod])
                        except ImportError:
                            pass                            
                else:
                    try:
                        __import__(delay_mod)
                    except ImportError:
                        pass
            delayed = []
                                        
            
    
    def _showhidden(self):
        """Show hidden variables"""
        self.__confmanager.showHidden()
        self.__init__(self.__confmanager) #restart
        
    
    def commit(self):
        """Update cloud with new settings.  
        
         .. warning::
            
            This will restart any active cloud instances, wiping mp/simulated jobs and setkey information
        """                
        import cloud        
        setattr(cloud,'__immutable', False)
        cloud.cloudinterface._setcloud(cloud, type=None)
        if hasattr(cloud,'mp'):
            setattr(cloud.mp,'__immutable', False)
            cloud.cloudinterface._setcloud(cloud.mp, type=None)
        
        #Reload cloud modules in correct order
        mods = cloud._modHook.mods[:]
        
        for modstr in mods:
            mod = sys.modules.get(modstr)
            if mod and modstr not in ['cloud.util.configmanager', 'cloud.cloudconfig']:
                try:
                    reload(mod)
                except ImportError:
                    pass       
        reload(cloud)        
        cloud._modHook.mods = mods #restore mods after it is wiped
        
    def __init__(self, confmanager, do_reload=False):
        backend = confmanager.backend
        self.__confmanager = confmanager
        
        def _set_prop(item):
            if hasattr(backend, item):
                typ = type(getattr(backend, option))
                if typ is type(None):
                    typ = None
            else:
                typ = None
            #print 'item %s has type %s' % (item, typ)
            def __inner__(self, value):
                if typ:
                    try:
                        k = typ(value)
                        setattr(backend,item, k)
                    except ValueError, e:
                        raise ValueError('Configuration option %s must have type %s.' % (option, typ.__name__))                    
                
            return __inner__
            
        def _get_prop(item):
            def __inner__(self):
                return getattr(backend, item)
            return __inner__
        
        import cloud
        ConfigSettings._loader(cloud.__path__[0] + os.sep ,'cloud.',do_reload)        
        for options in confmanager.sections.values():
            for option in options:
                prop = property(_get_prop(option), _set_prop(option), None, confmanager.optioncomment.get(ConfigManager.getCommentStr("",option)))
                setattr(self.__class__, option, prop)
                        
        
