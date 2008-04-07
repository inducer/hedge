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




#include "base.hpp"
#include "wrap_helpers.hpp"
#include <boost/python.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>




#ifdef USE_MPI




#include <boost/mpi.hpp>




namespace mpi = boost::mpi;
using namespace hedge;
using namespace boost::python;
using namespace boost::numeric::bindings;




namespace
{
  mpi::request isend_vector(
      mpi::communicator &comm,
      int dest, int tag,
      const py_vector &v)
  {
    return comm.isend(dest, tag, traits::vector_storage(v), traits::vector_size(v));
  }




  mpi::request irecv_vector(
      mpi::communicator &comm,
      int dest, int tag,
      py_vector v)
  {
    return comm.irecv(dest, tag, traits::vector_storage(v), traits::vector_size(v));
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

#ifdef USE_MPI
  def("isend_vector", isend_vector, 
      (arg("comm"), arg("dest"), arg("tag"), arg("vector")),
      with_custodian_and_ward_postcall<0,4>());
  def("irecv_vector", irecv_vector, 
      (arg("comm"), arg("source"), arg("tag"), arg("vector")),
      with_custodian_and_ward_postcall<0,4>());
#endif
}
