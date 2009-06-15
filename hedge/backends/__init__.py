"""Automated backend choosing."""

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




import hedge.discretization
import hedge.mesh




class RunContext(object):
    @property
    def rank(self):
        raise NotImplementedError

    @property
    def ranks(self):
        raise NotImplementedError

    @property
    def head_rank(self):
        raise NotImplementedError

    @property
    def is_head_rank(self):
        return self.rank == self.head_rank

    def distribute_mesh(self, mesh, partition=None):
        """Take the Mesh instance `mesh' and distribute it according to `partition'.

        If partition is an integer, invoke PyMetis to partition the mesh into this
        many parts, distributing over the first `partition' ranks.

        If partition is None, act as if partition was the integer corresponding
        to the current number of ranks on the job.

        If partition is not an integer, it must be a mapping from element number to
        rank. (A list or tuple of rank numbers will do, for example, or so will
        a full-blown dict.)

        Returns a mesh chunk.

        We deliberately do not define the term `mesh chunk'. The return value
        of this function is to be treated as opaque by the user, only to be
        used as an argument to L{make_discretization}().

        This routine may only be invoked on the head rank.
        """
        raise NotImplementedError

    def receive_mesh(self):
        """Wait for a mesh chunk to be sent by the head rank.

        We deliberately do not define the term `mesh chunk'. The return value
        of this function is to be treated as opaque by the user, only to be
        used as an argument to L{make_discretization}().

        This routine should only be invoked on non-head ranks.
        """

        raise NotImplementedError

    def make_discretization(self, mesh_data, *args, **kwargs):
        """Construct a Discretization instance.

        `mesh_data' is whatever gets returned from distribute_mesh and
        receive_mesh(). Any extra arguments are directly forwarded to
        the respective Discretization constructor.
        """
        raise NotImplementedError





class SerialRunContext(RunContext):
    communicator = None

    def __init__(self, discr_class):
        self.discr_class = discr_class

    @property
    def rank(self):
        return 0

    @property
    def ranks(self):
        return [0]

    @property
    def head_rank(self):
        return 0

    def distribute_mesh(self, mesh, partition=None):
        return mesh

    def make_discretization(self, mesh_data, *args, **kwargs):
        return self.discr_class(mesh_data, *args, **kwargs)




FEAT_MPI = "mpi"
FEAT_CUDA = "cuda"



def generate_features(allowed_features):
    if FEAT_MPI in allowed_features:
        import pytools.prefork
        pytools.prefork.enable_prefork()

        try:
            import boostmpi.autoinit
        except ImportError:
            pass
        else:
            import boostmpi as mpi
            if mpi.size > 1:
                yield FEAT_MPI

    if FEAT_CUDA in allowed_features:
        try:
            import pycuda
        except ImportError:
            have_cuda = False
        else:
            import pycuda.driver
            try:
                if pycuda.driver.Device.count():
                    yield FEAT_CUDA
            except pycuda.driver.LogicError:
                # pycuda not initialized--we'll give it the benefit of the doubt.
                yield FEAT_CUDA




def guess_run_context(allow=None):
    if allow is None:
        import sys

        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.startswith("--features="):
                allow = arg[arg.index("=")+1:].split(",")
                i += 1
            elif arg == "-f" and i+1 < len(sys.argv):
                allow = sys.argv[i+1].split(",")
                i += 2
            else:
                i += 1

        if allow is None:
            allow = []

    feat = list(generate_features(allow))

    if FEAT_CUDA in feat:
        from hedge.backends.cuda import Discretization as discr_class
    else:
        from hedge.backends.jit import Discretization as discr_class

    if FEAT_MPI in feat:
        from hedge.backends.mpi import MPIRunContext
        import boostmpi as mpi
        return MPIRunContext(mpi.world, discr_class)
    else:
        return SerialRunContext(discr_class)
