"""
Copyright (c) 2012 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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

"""
This code is based on a  simple XML-generator# Originally Lars Marius Garshol, September 1998
http://mail.python.org/pipermail/xml-sig/1998-September/000347.html
Changes by Uche Ogbuji April 2003
 *  unicode support: accept encoding argument and use Python codecs
    for correct character output
 *  switch from deprecated string module to string methods
 *  use PEP 8 style
 *  Further modifications by PiCloud

Please see Ogbuji's article for background: http://www.xml.com/pub/a/2003/04/09/py-xml.html
Ogbuji has given permission to modify and release this code under LGPL
"""
 

import sys
import codecs

class XmlWriter:

    def __init__(self, out=sys.stdout, encoding="utf-8", indent=u"  ", header=True):
        """
        out      - a stream for the output
        encoding - an encoding used to wrap the output for unicode
        indent   - white space used for indentation
        """
        wrapper = codecs.lookup(encoding)[3]
        self.out = wrapper(out)
        self.stack = []
        self.indent = indent
        if header:
            self.out.write(u'<?xml version="1.0" encoding="%s"?>\n' \
                       % encoding)
        self.needClose = False

    def close(self):
        if self.out is not sys.stdout:
            self.out.close()

    def doctype(self, root, pubid, sysid):
        """
        Create a document type declaration (no internal subset)
        """
        if pubid == None:
            self.out.write(
                u"<!DOCTYPE %s SYSTEM '%s'>\n" % (root, sysid))
        else:
            self.out.write(
                u"<!DOCTYPE %s PUBLIC '%s' '%s'>\n" \
                % (root, pubid, sysid))
    
    def _closeIfNeeded(self):
        if self.needClose:
            self.out.write(u">")
        self.needClose = False
        
    def comment(self, cmt, attrs={}):
        """
        Create a comment with attributes
        """
        self._closeIfNeeded()
        if self.stack:
            self.out.write('\n')
        self.out.write('<!-- ')
        self.out.write(cmt)
        for (a, v) in attrs.items():
            try:
                self.out.write(u" %s='%s'" % (a, self.__escape_attr(v)))
            except UnicodeDecodeError:
                continue        
        self.out.write(' -->')
        if not self.stack:
            self.out.write('\n')

        
    def push(self, elem, attrs={}):        
        """
        Create an element which will have child elements
        """
        self._closeIfNeeded()
        if self.stack:
            self.out.write('\n')
        self.__indent()
        try:
            self.out.write("<" + elem)
        except UnicodeDecodeError:
            elem = 'UNPRINTABLE_OBJECT'
            self.out.write("<" + elem)  
        for (a, v) in attrs.items():
            try:
                self.out.write(u" %s='%s'" % (a, self.__escape_attr(v)))
            except UnicodeDecodeError:
                continue
        if self.stack:
            self.needClose = True
        else:
            self.out.write(u">")
        self.stack.append(elem)

    def elem(self, elem, content, attrs={}):
        """
        Create an element with text content only
        """
        self._closeIfNeeded()
        if self.stack:
            self.out.write('\n')
        self.__indent()
        self.out.write(u"<" + elem)
        for (a, v) in attrs.items():
            self.out.write(u" %s='%s'" % (a, self.__escape_attr(v)))
        self.out.write(u">%s</%s>" \
                       % (self.__escape_cont(content), elem))
        if not self.stack:
            self.out.write('\n')

    def empty(self, elem, attrs={}):
        """
        Create an empty element
        """
        self._closeIfNeeded()
        if self.stack:
            self.out.write('\n')
        self.__indent()
        self.out.write(u"<"+elem)
        for a in attrs.items():
            self.out.write(u" %s='%s'" % a)
        self.out.write(u"/>")
        if not self.stack:
            self.out.write('\n')
        
    def pop(self):
        """
        Close an element started with the push() method
        """
        elem=self.stack[-1]
        del self.stack[-1]

        if self.needClose: #means we never pushed a subelement onto stack:
            self.out.write(u"/>")
            self.needClose = False                        
        else:
            self.out.write('\n')        
            self.__indent()        
            self.out.write(u"</%s>" % elem)
        #if outer level:
        if not self.stack:
            self.out.write('\n')
        
    def flush(self):
        self.out.flush()
    
    def __indent(self):
        self.out.write(self.indent * (len(self.stack) * 2))
    
    def __escape_cont(self, text):
        return text.replace(u"&", u"&amp;")\
               .replace(u"<", u"&lt;")

    def __escape_attr(self, text):
        return text.replace(u"&", u"&amp;") \
               .replace(u"'", u"&apos;").replace(u"<", u"&lt;")
               
"""
Stack Writer
Will only write a single downward path in the xml tree
"""
class XmlStackWriter:

    def __init__(self, xmlwriter):
        """
        xmlwriter      -  real xmlwriter
        """
        self.xmlwriter = xmlwriter
        self.writecmd = []
        self.extraEl = False
    
    def addElement(self, node):        
        if self.extraEl:
            self.writecmd[-1] = node
        else:
            self.writecmd.append(node)
            self.extraEl = True       
        
    def comment(self, cmt, attrs={}):
        newcmd = (self.xmlwriter.comment,(cmt,attrs))
        self.addElement(newcmd)

        
    def push(self, elem, attrs={}):        
        newcmd = (self.xmlwriter.push,(elem,attrs))
        self.addElement(newcmd)
        self.extraEl = False


    def elem(self, elem, content, attrs={}):
        newcmd = (self.xmlwriter.elem,(elem,content,attrs))
        self.addElement(newcmd)
                    
    def empty(self, elem, attrs={}):
        newcmd = (self.xmlwriter.empty,(elem,attrs))
        self.addElement(newcmd)        
        
    def pop(self):
        if self.extraEl:
            self.writecmd.pop()        
        self.writecmd.pop()        
        self.extraEl = False
        
    def flush(self):
        """
        Flush to xml file
        """        
        for cmd, args in self.writecmd:                     
            cmd(*args)
        self.xmlwriter._closeIfNeeded() #prettification hack

"""
Reading:
"""

import xml.parsers.expat

"""
when you get to a tag:
Cases:
1. If tag has the same name as a class (class.xmltagname()) or if it has a !type attribute! that has the same name as a class (class.xmltagname(): instantiate the class
    - If parent is None, assign class to global list
    - If parent is a class, assign to its member variable with the correct name specified by the !name attribute! (default to tag name)
    - If the parent is a list, create a new object and append it
    - If the parent is a dictionary, create a new key/value pair in the dictionary
2. If tag does not have the same name as a class:
    - If parent is None?
    - If the parent is a class, it must have the same tag name or !name attribute! as a member of the parent class (if there is no parent, throw an exception)
    - If the parent is a dictionary, and the tag has a value, add it as a key/value to the dictionary
    - If the parent is a dictionary, and the tag does not have a value, then it is in limbo mode
    - If the parent is a list, and the tag has a value... (not currently specified)
    - If the parent is not yet specified, its a list, unless there is a value attribute

if tag does not have the same name as a class, it can be either a list or a dict


"""

class XmlDataParser:

    path = []
    objpath = []

    root = None

    parser = None
    tagclasses = {}
    
    lasttagname = None

    def __init__(self):
        self.parser = xml.parsers.expat.ParserCreate()

        self.parser.StartElementHandler = self.start_element
        self.parser.EndElementHandler = self.end_element
        self.parser.CharacterDataHandler = self.char_data

    def addTagClass(self, cls):
        self.tagclasses[cls.xmltagname()] = cls

    def parse(self, filename):
        file = open(filename)
        lines = file.readlines()
        xmltext = ''
        for line in lines:
            xmltext += line.strip()        
        self.parser.Parse(xmltext)
        
        #print "root is " + str(self.root)
        return self.root

    # 3 handler functions
    def start_element(self, name, attrs):

        if len(self.objpath) > 0:
            parent = self.objpath[len(self.objpath)-1]
        else:
            parent = None

        if 'type' in attrs:
            tagtype = attrs['type']
        else:
            tagtype = name

        if 'name' in attrs:
            tagname = attrs['name']
        else:
            tagname = name
        
        self.lasttagname = tagname

        #print 'Entering', name
        #print 'OBJPATH', str(self.objpath)
        #print 'PATH', str(self.path)
        #print 'ROOT:', str(self.root)
        #print 'INFO:', tagtype, tagname
        #print 'PARENT:', parent

        self.path.append(name)

        #if tag has the same name as a class
        if tagtype in self.tagclasses:
            #print 'tagtype in tagclass'
            #newobj = self.tagclasses[tagtype]()
            targetclass = self.tagclasses[tagtype]
            newobj = targetclass()

            if parent is None:
                #print '1'
                self.root = newobj
            elif type(parent) is list:
                #print '2'
                parent.append(newobj)
            elif type(parent) is dict:
                #print '3'
                pass
            else:
                #print '4'
                if hasattr(parent, tagname):
                    setattr(parent, tagname, newobj)
                else:
                    raise Exception('Parent object does not have attribute ' + str(tagname) )
            self.objpath.append(newobj)

        else:
            #print 'tagtype not in tagclass'
            if parent is None:
                #print '1'
                self.root = []
                self.objpath.append(self.root)
            elif type(parent) is dict and 'value' in attrs:
                #print '2'
                parent[tagname] = self.convertValueToBestType(attrs['value'])
                self.objpath.append('N/A')
            elif type(parent) is dict and 'value' not in attrs:
                raise Exception('Parent is dict, but child has no value attribute')
            elif type(parent) is list:                                
                raise Exception('Not really supported... ' + str(parent) + " with " + str(tagname))
            else:
                #print '3'

                if hasattr(parent, tagname) and 'value' in attrs:
                    self.objpath.append('N/A')
                    setattr(parent, tagname, self.convertValueToBestType(attrs['value']))
                elif hasattr(parent, tagname):
                    self.objpath.append(getattr(parent, tagname))
                else:
                    raise Exception('Parent object does not have attribute ' + str(tagname))

        #print ''

    # assume value is passed is as a string
    # as it would be when being parsed in an xml file
    def convertValueToBestType(self, value):
        #print value, type(value)
        if not isinstance(value, basestring):
            raise Exception('convertValueToCorrectType given a non-string input')

        try:
            floatval = float(value)
            if floatval % 1.0 == 0.0:
                return int(value)
            else:
                return floatval
        except ValueError, e:
            lowerval = value.lower()
            if lowerval == "true":
                return True
            elif lowerval == "false":
                return False

            return value

    def end_element(self, name):
        #print "Leaving", name
        self.path.pop()
        self.objpath.pop()
        #if len(self.objpath) > 0:
        #    parent = self.objpath[len(self.objpath)-1]
        #    #print parent
        #print '' 

    def char_data(self, data):
        if len(self.objpath) > 1:
            parent = self.objpath[len(self.objpath)-2]
        else:
            parent = None
        #print 'Character data:', repr(data)
        if hasattr(parent, self.lasttagname):            
            setattr(parent, self.lasttagname, self.convertValueToBestType(str(data)))
        else:        
            raise Exception("cannot handle data < " + repr(data) + "> inside " + str(self.lasttagname))


#p.Parse("""<?xml version="1.0"?>
#<parent id="top"><bob id="asfaf" /><child1 name="paul">Text goes here</child1>
#<child2 name="fred">More text</child2>
#</parent>""", 1)


