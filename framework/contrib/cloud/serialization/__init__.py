"""
Serialization submodule
Responsible for all cloud related serialization routines

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

from .serializationhandlers import Serializer, CloudSerializer, DebugSerializer, Deserializer
from .report import SerializationReport

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

def serialize(obj, needsPyCloudSerializer = False, useDebugSerializer = False):
    """Pickle obj into a string.  
    If needsPyCloudSerializer is set, additional types are allowed at the cost of speed
    If useDebugSerializer is set, exceptions are more detailed, again at the cost of speed
    """
    
    if not needsPyCloudSerializer:
        return Serializer(obj).run_serialization()
    elif not useDebugSerializer:
        return CloudSerializer(obj).run_serialization()
    else:
        return DebugSerializer(obj).run_serialization()
    
def deserialize(str):
    """Deserialize the serialized object described by string str"""
    return Deserializer(str).deserializedObj