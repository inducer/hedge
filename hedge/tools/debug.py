"""Debugging aids."""

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



from pytools import memoize
import numpy




def wait_for_keypress(discr):
    """MPI-aware keypress wait"""
    try:
        comm = discr.parallel_discr.context.communicator
    except AttributeError:
        raw_input("[Enter]")
    else:
        if comm.rank == 0:
            # OpenMPI connects mpirun's stdin to rank 0's stdin.
            print "[Enter]"
            raw_input()

        from boostmpi import broadcast
        broadcast(comm, value=0, root=0)




def get_rank(discr):
    """Rank query that works with and without MPI active."""
    try:
        comm = discr.parallel_discr.context.communicator
    except AttributeError:
        return 0
    else:
        return comm.rank




def typedump(value, max_seq=5, special_handlers={}):
    from pytools import typedump
    special_handlers = special_handlers.copy()
    special_handlers.update({
        numpy.ndarray: lambda x: "array(%s, %s)" % (len(x.shape), x.dtype)
        })
    return typedump(value, max_seq, special_handlers)




from pytools.debug import (
        make_unique_filesystem_object,
        get_run_debug_directory)


def open_unique_debug_file(stem, extension=""):
    from pytools.debug import (
            open_unique_debug_file as
            open_unique_debug_file_base)
    return open_unique_debug_file_base(stem, extension)[0]




def mem_checkpoint(name=None):
    """Invoke the garbage collector and wait for a keypress."""

    import gc
    gc.collect()
    if name:
        raw_input("%s -- hit Enter:" % name)
    else:
        raw_input("Enter:")
