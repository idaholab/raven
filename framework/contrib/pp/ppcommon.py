# Parallel Python Software: http://www.parallelpython.com
# Copyright (c) 2005-2012, Vitalii Vanovschi
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the author nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
"""
Parallel Python Software, Execution Server

http://www.parallelpython.com - updates, documentation, examples and support
forums
"""

import threading

copyright = "Copyright (c) 2005-2012 Vitalii Vanovschi. All rights reserved"
version = "1.6.4"

def start_thread(name,  target,  args=(),  kwargs={},  daemon=True):
    """Starts a thread"""
    thread = threading.Thread(name=name,  target=target, args=args,  kwargs=kwargs)
    thread.daemon = daemon
    thread.start()
    return thread


def get_class_hierarchy(clazz):
    classes = []
    if clazz is type(object()):
        return classes
    for base_class in clazz.__bases__:
        classes.extend(get_class_hierarchy(base_class))
    classes.append(clazz)
    return classes


def is_not_imported(arg, modules):
    args_module = str(arg.__module__)
    for module in modules:
        if args_module == module or args_module.startswith(module + "."):
            return False
    return True

# Parallel Python Software: http://www.parallelpython.com
