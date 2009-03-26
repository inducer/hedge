# This is a wrapper around boostmpi to allow MPI children to fork.
# See pytools.prefork for a full discussion of the issue.
# If you're looking for hedge's MPI functionality, see
# src/python/backends/mpi.

import pytools.prefork
pytools.prefork.enable_prefork()
from boostmpi import *
