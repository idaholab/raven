"""
Caching objects

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


from __future__ import with_statement 
import threading
import weakref
import collections

from .xrange_helper import maybe_xrange_iter 

#unique object to identify when attribute not present
#By design this is designed to evaluate to False
_NO_ATTRIBUTE = list()

class JobCacheItem(object):
    """Stub cache item.
    Attributes are dynamically added by the managers
    
    USAGE NOTE: None is reserved for a non-existent field.  One cannot cache None
    """
    
    def update(self, adict):
        """Update attributes with adict"""
        for key, val in adict.items():
            setattr(self,key,val)

class JobAbstractCacheManager(object):
    """
    Abstract Cache manager that does not cache
    """
    
    def getCached(self, jids, fields):
        """
        Request cached information about jids        
        return sequence of respective jid->field mapping for each field respectively
        """        
        return  {}
    
    def putCached(self, jid, **kwargs):
        """
        Cache a jid with certain properties
        """           
        pass
    
    def deleteCached(self, jids):
        """
        Remove these jids from the cache
        """
        pass        

class JobCacheManager(JobAbstractCacheManager):
    """
    Simple manager for dealing with cloud cache
    """
    
    jobMap = None #maps jids to cache
    cacheLock = None 
    myItem = JobCacheItem  #type of cache item to generate
    
    def __init__(self):        
        self.jobMap = {} #maps jids to cache
        self.cacheLock = threading.RLock()        
        
    def _getJob(self, jid):
        return self.jobMap.get(jid, None)
    
    def getCached(self, jids, fields):
        """
        Request cached information about jids        
        return dictionary mapping jid to dictionary describing cached fields
        If no cached fields, jid will not appear in dictionary
        """
        outdict = {}
                       
        for jid in maybe_xrange_iter(jids):
            job = self._getJob(jid)
            if job:
                jf = {}
                for field in fields:
                    attr = getattr(job,field,_NO_ATTRIBUTE)
                    if attr != _NO_ATTRIBUTE:                 
                        jf[field] = attr
                if jf:
                    outdict[jid]  = jf    
        return outdict
    
    def _deleteJob(self, jid):
        try:
            del self.jobMap[jid]
        except KeyError:
            pass
        
    
    def deleteCached(self, jids):
        """
        Remove these jids from the cache
        """
        with self.cacheLock:
            for jid in jids:
                self._deleteJob(jid)
    
    def _addJob(self, jid, item):
        self.jobMap[jid] = item
        return item
            
    def putCached(self, jid, **kwargs):    
        """
        Cache a jid with certain properties
        """    
        with self.cacheLock:    
            job = self._getJob(jid)
            if not job:
                job = self.myItem()
                needAdd = True
            else:
                needAdd = False
            job.update(kwargs)
            if needAdd:
                return self._addJob(jid, job)
            return job

class JobFiniteCacheManager(JobCacheManager):
    """
    Regular JobCacheManager which utilizes clock algorithm to limit number of elements in cache
    """
    jobClock = None #clock algorithm to push jobs out
    clockHand = 0
    numDeleted = 0 #indicates number of non-purged jobClock entries

    def __init__(self, cacheSize):
        JobCacheManager.__init__(self)        
        self.cacheSize = cacheSize
        self.jobClock = [] #clock algorithm to push jobs out
        self.clockHand = 0    
        
    def _replaceNotifier(self, oldjid):
        """Notification when oldjid is to be replaced"""
        self.numDeleted-=1
    
    def _replace(self, newjid):
        """Find a spot for newjid"""
        sz = len(self.jobClock)    
        while (True):
            self.clockHand=(1+self.clockHand)%sz
            self.clockHand%=sz
            jid = self.jobClock[self.clockHand]
            if self.numDeleted:  
                #replace deleted entries first
                #technically it is wrong to move the clockHand, but delete is rare
                if jid not in self.jobMap:
                    self.numDeleted -=1
                    self.jobClock[self.clockHand] = newjid                                       
                    return                    
            elif getattr(self.jobMap[jid],'clockBit',0) == 0:
                del self.jobMap[jid]  #don't use _deleteJob as it would increment numDeleted
                self.jobClock[self.clockHand] = newjid
                return
            else:
                self.jobMap[jid].clockBit = 0 
    
    def _getJob(self, jid):
        item = self.jobMap.get(jid, None)
        if item:
            item.clockBit = 1        
        return item
    
    def _addJob(self, jid, item):
        self.jobMap[jid] = item
        if len(self.jobMap) > self.cacheSize:
            self._replace(jid)
        else:
            self.jobClock.append(jid)
        return item

    def _deleteJob(self, jid):
        try:
            del self.jobMap[jid]
        except KeyError:
            pass
        else:
            self.numDeleted+=1


class JobFiniteSizeCacheManager(JobFiniteCacheManager):
    """
    A JobFiniteCacheManager limited by total cache size
    NOTE: This can only store objects that define __len__ (e.g. serialized results)
        __len__ is considered the object's size
        Size of actual cache items is ignored in size calculation
    cacheSize is interpreted as maximum space strings can take
    
    Clock algorithm is still used.
    Items taking >= 0.5*total space are not cached
    
    A double-sided queue is used for the clock     
    """
    
    mySize = 0 #size this cache is consuming
    trackedAttr = "" #Name of attribute of JobCacheItem whose size is being tracked
    
    def __init__(self, cacheSize, trackedAttr):
        JobCacheManager.__init__(self)        
        self.cacheSize = cacheSize
        self.jobClock = collections.deque() 
        self.trackedAttr = trackedAttr
    
    def _makeSpace(self, extraSize):
        """Clear out space"""
        sz = len(self.jobClock)    
        while (self.mySize + extraSize >= self.cacheSize):
            jid = self.jobClock.pop()
            if jid in self.jobMap:      #if deleted, no longer here                    
                if getattr(self.jobMap[jid],'clockBit',0) == 0:
                    self._deleteJob(jid)
                else:
                    self.jobMap[jid].clockBit = 0
                    self.jobClock.appendleft(jid) 
        
    def _deleteJob(self, jid):
        try:
            item = self.jobMap[jid]
            del self.jobMap[jid]
        except KeyError:
            pass
        else:
            self.mySize -= len(getattr(item, self.trackedAttr))
            #jobclock entry is deleted eventually by _makeSpace
    
    def _addJob(self, jid, item):
        
        sz = len(getattr(item, self.trackedAttr))
        if 2*sz >= self.cacheSize: #ignore items that are too large
            return None 
        
        self._makeSpace(sz)
        
        self.jobMap[jid] = item
        self.jobClock.appendleft(jid)
        
        self.mySize += sz
        return item            

class JobFiniteDoubleCacheManager(JobFiniteCacheManager):
    """
    A JobFiniteCacheManager that has weakrefs to items in a JobFiniteSizeCacheManager (childManager)
    weakrefs are used to point to a potentially large data structure that can be ejected by the
        childManager. We still hold the other pieces of data though
    """
    
    weakRefAttr = None  #Item that should be moved to child manager
    childManager = None #child manager that tracks size
    
    _constructedItems = {}  #static dictionary mapping to constructed classes
    
    @classmethod
    def _constructJobItemType(cls, trackedAttr):
        """Construct a JobCacheItem subclass that has weakRefAttr 
        as a property that dereferences the weakref"""
        
        #check cache first
        try:
            return cls._constructedItems[trackedAttr]
        except KeyError:
            pass
        
        hiddenAttr = '_' + trackedAttr #hide weakref
        
        def _generateProp():
            def getProp(self):
                wrefItem = getattr(self, hiddenAttr)
                if not wrefItem:
                    return None                                
                wrefItem = wrefItem() #dereference weakref
                if not wrefItem:
                    return _NO_ATTRIBUTE              
                wrefItem.clockBit = 1 #update clock of item                
                return getattr(wrefItem, trackedAttr) 
            
            #TODO: Setprop (would need access to cache manager)            
            return property(getProp, doc="Access %s through weakref" % trackedAttr)
        
        dct = {trackedAttr: _generateProp()}
        JobItemType = type('JobCacheItem' + trackedAttr, (JobCacheItem,), dct)
        
        cls._constructedItems[trackedAttr] = JobItemType
        return JobItemType
        
        
    
    def __init__(self, cacheSize, childManager):
        JobFiniteCacheManager.__init__(self, cacheSize)
        self.childManager = childManager 
        tracked = childManager.trackedAttr
        self.myItem = self.__class__._constructJobItemType(tracked)
        self.weakRefAttr = tracked  #weakref item is the child's trackedAttr        
        
                        
    def putCached(self, jid, **kwargs):
        #Generate weakref if kwargs includes target        
        if self.weakRefAttr in kwargs:
            newItem = self.childManager.putCached(jid,**{self.weakRefAttr: kwargs[self.weakRefAttr]})
            if newItem: #child is allowed to reject
                #We must make a weakref to the item, as weakrefs to strings are not allowed
                kwargs['_' + self.weakRefAttr] = weakref.ref(newItem)
            del kwargs[self.weakRefAttr] #don't let parent put in cache            
        return JobFiniteCacheManager.putCached(self, jid, **kwargs)
        
    def _deleteJob(self, jid):
        #Delete child's result
        self.childManager.deleteCached([jid])
        JobFiniteCacheManager._deleteJob(self,jid)
        
    def _replaceNotifier(self, oldjid):
        """Notification when oldjid is to be replaced"""
        JobFiniteCacheManager._replaceNotifier(self, oldjid)
        #wipe child entry:
        JobFiniteCacheManager.deleteCached([oldjid])       
        
        
    
    
