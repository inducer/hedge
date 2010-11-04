"""Debugging aids."""

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
        get_run_debug_directory,
        open_unique_debug_file)




def mem_checkpoint(name=None):
    """Invoke the garbage collector and wait for a keypress."""

    import gc
    gc.collect()
    if name:
        raw_input("%s -- hit Enter:" % name)
    else:
        raw_input("Enter:")
