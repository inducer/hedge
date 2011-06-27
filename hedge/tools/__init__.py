"""Miscellaneous helper facilities."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""





import numpy
import hedge._internal

# don't import stuff from here--this is purely for backward-compatibility
from pytools.obj_array import *
from hedge.optemplate.tools import *
from hedge.tools.mathematics import *
from hedge.tools.linalg import *
from hedge.tools.convergence import *
from hedge.tools.flops import *
from hedge.tools.debug import *
from hedge.tools.indexing import *
from hedge.tools.affine import *
from hedge.flux.tools import *




# small utilities -------------------------------------------------------------
def is_zero(x):
    # DO NOT try to replace this with an attempted '== 0' comparison.
    # This will become an elementwise numpy operation and not do what
    # you want.

    if isinstance(x, (int, complex, float, numpy.number)):
        return x == 0
    else:
        return False




class Closable(object):
    def __init__(self):
        self.is_closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self.is_closed:
            # even if close attempt fails, consider ourselves closed still
            try:
                self.do_close()
            finally:
                self.is_closed = True




def reverse_lookup_table(lut):
    result = [None] * len(lut)
    for key, value in enumerate(lut):
        result[value] = key
    return result
