"""Futures, i.e. lazy evaluation."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

__license__ = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see U{http://www.gnu.org/licenses/}.
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
