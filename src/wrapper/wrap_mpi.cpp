// Hedge - the Hybrid'n'Easy DG Environment
// Copyright (C) 2007 Andreas Kloeckner
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.




#include "wrap_helpers.hpp"
#include <boost/python.hpp>




#ifdef USE_MPI

#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
namespace py = boost::python;




namespace
{
  mpi::request isend_buffer(
      mpi::communicator &comm,
      int dest, int tag,
      const py::object obj)
  {
    const void *buf;
    Py_ssize_t len;
    PyObject_AsReadBuffer(obj.ptr(), &buf, &len);
    return comm.isend(dest, tag, reinterpret_cast<const char *>(buf), len);
  }




  mpi::request irecv_buffer(
      mpi::communicator &comm,
      int dest, int tag,
      const py::object obj)
  {
    void *buf;
    Py_ssize_t len;
    PyObject_AsWriteBuffer(obj.ptr(), &buf, &len);
    return comm.irecv(dest, tag, reinterpret_cast<char *>(buf), len);
  }
}




#endif




namespace
{
  bool have_mpi()
  {
#ifdef USE_MPI
    return true;
#else
    return false;
#endif
  }
}




void hedge_expose_mpi()
{
  DEF_SIMPLE_FUNCTION(have_mpi);

  using py::args;

#ifdef USE_MPI
  py::def("isend_buffer", isend_buffer, 
      args("comm", "dest", "tag", "vector"),
      py::with_custodian_and_ward_postcall<0,4>());
  py::def("irecv_buffer", irecv_buffer, 
      args("comm", "source", "tag", "vector"),
      py::with_custodian_and_ward_postcall<0,4>());
#endif
}
