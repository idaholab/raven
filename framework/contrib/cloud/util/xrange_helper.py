"""
XRange serialization routines

Note: This code is in desperate need of cleanup. But options are limited.
Some bad API choices were made 2 years ago, and cannot be reverted

The code deals with several types.
The PiecewiseXrange is a list subclass that has xranges internally
Example:
[1,3,xrange(5,9), 20]
This defines the list [1,3,5,6,7,8,20] more compactly

This also deals with JSONable PiecewiseXrange. These encodings can then be encoded by JSON. They look like:
[1,3,['xrange', 5, 9, 1], 20]

Note:
Single integers are also considered PiecewiseXrange (e.g. 9 is a PiecewiseXrange)
Single xranges are also PiecewiseXrange. (e.g. xrange(10) is a PiecewiseXrange)


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


def xrange_params(xrangeobj):
    """Returns a 3 element tuple describing the xrange start, step, and len respectively
    
    Note: Only guarentees that elements of xrange are the same. parameters may be different.    
    e.g. xrange(1,1) is interpretted as xrange(0,0); both behave the same though w/ iteration
    """
    
    xrange_len = len(xrangeobj)
    if not xrange_len: #empty
        return (0,1,0) 
    start = xrangeobj[0]
    if xrange_len == 1: #one element
        return start, 1, 1 
    return (start, xrangeobj[1] - xrangeobj[0], xrange_len) 
    

"""Encoding (generally for placing within a JSON object):"""
    
def encode_maybe_xrange(obj, allowLists = True):
    """If obj is a picewiseXrange, make it a JSONable PiecewiseXrange
    Else return object
    If allowLists is False only an xrange can be encoded
    If true, PiecewiseXranges can be encoded 
    """
    if isinstance(obj, xrange):        
        return ['xrange'] + list(xrange_params(obj))
    if allowLists and isinstance(obj, list):
        return [encode_maybe_xrange(elm, allowLists = False) for elm in obj]
    return obj

def decode_maybe_xrange(obj, allowLists = True):
    """Decode an object that may be a JSONable piecewise xrange, into 
    a regular PiecewiseXrange
    Else return object"""
    if isinstance(obj, list):
        if len(obj) == 4 and obj[0] == 'xrange':  #an xrange object
            return xrange(obj[1], obj[1] + obj[2]*obj[3], obj[2])            
        elif allowLists: #decode internal xranges
            return [decode_maybe_xrange(elm, allowLists = False) for elm in obj]
    return obj

def decode_xrange_list(obj):
    """First call decode_maybe_xrange(obj)
    Then flatten the list, so no xranges are contained"""     
    
    obj =  decode_maybe_xrange(obj, allowLists = False) #outer layer might be an xrange
    if isinstance(obj, xrange):
        return list(obj) #convert to list
    outlst = []
    for elem in obj:
        if isinstance(elem, list) and len(elem) == 4 and elem[0] == 'xrange':
            outlst.extend(decode_maybe_xrange(elem, allowLists = False))
        else:
            outlst.append(elem)
    return outlst

"""Support for piece-wise xranges - effectively a list of xranges"""
class PiecewiseXrange(list):
    """A PiecewiseXrange is best explained with an example:
    
    [1,3,xrange(5,9), 20] is a PiecewiseXrange that defines    
    
    [1,3,5,6,7,8,20] more compactly
    
    This class exists to provide the my_iter method which iterates the elements defined.
    in example, myiter (in xrangeMode) would return:
    1,3,5,6,7,8,20 
    
    However, __iter__ returns:
    1,3,xrange(5,9), 20
    """
    
    def __init__(self):
        self._xrangeMode = False
        
    def to_xrange_mode(self):
        """Turn on xrange iterator"""
        self._xrangeMode = True
    
    def my_iter(self):
        """Ideally, this would overload __iter__, but that presents massive pickling issues"""
        if self._xrangeMode:
            return xrange_iter(self)
        else:
            return self.__iter__()

def xrange_iter(iterable):  #use a generator to iterate
    for elm in iterable.__iter__():
        if isinstance(elm, xrange):
            for subelm in elm:
                yield subelm
        else:
            yield elm

def maybe_xrange_iter(lst, coerce = True):
    """if lst is a PiecewiseXrange or coerce, iterate as though lst is a piecewise xrange
    otherwise, normal lst iteration 
    """
    if isinstance(lst, PiecewiseXrange):
        return lst.my_iter()
    elif coerce:
        return xrange_iter(lst)
    else:         
        return lst.__iter__()
        

def filter_xrange_list(func, xrange_list):
    """ Input is a PiecewiseXrange
        
        Returns a reasonably compact PiecewiseXrange consisting of all elements in the input where         
        where func(elem) evaluates to True
        
        Behavior notes:
        If an xrange reduces to one element, it is replaced by an integer
        e.g. xrange(n,n+1) = n
    """
    
    if not hasattr(xrange_list,'__iter__') or isinstance(xrange_list, xrange):
        xrange_list = [xrange_list]
    
    single_range = 2 #if > 0, then outList is just a single xrange
    no_xrange_output = True #outList has no xranges inserted
    
    outList = PiecewiseXrange()
    for elm in xrange_list:
        if isinstance(elm, (int, long)):  #elm is 
            if func(elm):
                outList.append(elm)
                single_range = 0 #individual elements present - so not single xrange
        elif isinstance(elm, xrange):            
            step = xrange_params(elm)[1]
            basenum = None
            for num in elm: #iterate through xrange
                if func(num):
                    if basenum == None:
                        basenum = num                        
                else:
                    if basenum != None: #push back xrange
                        if num - step == basenum: #only one element: push an integer
                            outList.append(basenum)
                            single_range = 0
                        else:
                            outList.append(xrange(basenum, num, step))
                            single_range-=1
                            no_xrange_output = False
                        basenum = None
            if basenum != None: #cleanup
                num+=step
                if num - step == basenum: #only one element: push an integer
                    outList.append(basenum)
                    single_range = 0
                else:
                    outList.append(xrange(basenum, num, step))
                    single_range-=1
                    no_xrange_output = False
        else:
            raise TypeError('%s (type %s) is not of type int, long or xrange' % (elm, type(elm)))
        
    if outList:
        if not no_xrange_output:
            if single_range > 0: #only one xrange appended - just return it
                return outList[0]
            else:
                outList.to_xrange_mode()
    return outList

def iterate_xrange_limit(pwx, limit):
    """    
    Split PiecewiseXrange *pwx* into individual pieceWiseXranges containing no more than limit elements 
    Iterator returns individual partitions
    """
    if not pwx: #empty list/non case
        return
    
    if isinstance(pwx, xrange):
        if len(pwx) <= limit:
            yield pwx
            return
        #needs to parition
        pwx = [pwx]
        
    #main algorithm
    outlist = []
    cnt = 0 #number of items appended
    for xr in pwx:
        if isinstance(xr, (int, long)):
            outlist.append(xr)
            cnt+=1
            if cnt >= limit:
                yield outlist
                outlist = []
                cnt =0
        elif isinstance(xr, xrange):
            while True:
                if len(xr) + cnt <= limit:
                    outlist.append(xr)
                    cnt+=len(xr)
                    break
                else: #break apart xrange
                    allowed = limit - cnt
                    start, step, xl_len = xrange_params(xr)
                    breakpoint = start+ step*allowed
                    to_app = xrange(start, breakpoint, step)
                    outlist.append(to_app)
                    yield outlist
                    outlist = []
                    cnt=0
                    xr = xrange(breakpoint,breakpoint+(xl_len-allowed)*step, step)
            if len(outlist) >= limit:
                yield outlist
                outlist = []  
                
        else:
            raise TypeError('%s (type %s) is not of type int, long or xrange' % (xr, type(xr)))
    if outlist:
        yield outlist    
    
"""
Quick unit test
(TODO: Move to unit tests)
"""

if __name__ == '__main__':
    
    for m in [xrange(0,10,2), xrange(0,1), xrange(0), xrange(0,-2), xrange(0,6), xrange(0,7,3)]:
        me = encode_maybe_xrange(m)
        n = decode_maybe_xrange(me)
        print m, me, n
        
        #split
        print list(iterate_xrange_limit(m,1000))
        print list(iterate_xrange_limit(m,3))
        print [list(x[0]) if type(x) is list else x for x in iterate_xrange_limit(m,3)]
    
    print 'filters!'
    xrl = filter_xrange_list(lambda x: x%10, xrange(10,100))
    print xrl
    
    print 'it limits!'
    lmt = list(iterate_xrange_limit(xrl,15))
    print 'chunks of 15', lmt
    
    lmt_read = [[list(x) for x in y] for y in lmt]
    lmt2 = [reduce(lambda x, y: x+y, x) for x in lmt_read] 
    print 'still 15', lmt2
    
    lenz = [len(x) for x in lmt2]
    print lenz
    
    print list(iterate_xrange_limit(xrange(25), 13))
    it = iterate_xrange_limit(xrange(25), 13)
    print 'iterate!'
    try:
        while True:
            v = it.next()
            print type(v), v
    except StopIteration:
        pass
        
        