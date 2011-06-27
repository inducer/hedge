"""Futures, i.e. lazy evaluation."""

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




class Future(object):
    """An abstract interface definition for futures.

    See http://en.wikipedia.org/wiki/Future_(programming)
    """
    def is_ready(self):
        raise NotImplementedError(self.__class__)

    def __call__(self):
        raise NotImplementedError(self.__class__)




class ImmediateFuture(Future):
    """A non-future that immediately has a value available."""
    def __init__(self, value):
        self.value = value

    def is_ready(self):
        return True

    def __call__(self):
        return self.value




class NestedFuture(Future):
    """A future that combines two sub-futures into one."""
    def __init__(self, outer_future_factory, inner_future):
        self.outer_future_factory = outer_future_factory
        self.inner_future = inner_future
        self.outer_future = None

    def is_ready(self):
        if self.inner_future.is_ready():
            self.outer_future = self.outer_future_factory(self.inner_future())
            self.is_ready = self.outer_future.is_ready()
            return self.is_ready()
        else:
            return False

    def __call__(self):
        if self.outer_future is None:
            return self.outer_future_factory(self.inner_future())()
        else:
            return self.outer_future()
