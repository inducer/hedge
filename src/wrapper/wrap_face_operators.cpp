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




#include <boost/python.hpp>
#include "op_target.hpp"
#include "face_operators.hpp"
#include "wrap_helpers.hpp"




using namespace boost::python;
using namespace hedge;
namespace ublas = boost::numeric::ublas;




namespace
{
  template <class A, class B>
  std::pair<A, B> extract_pair(const object &obj)
  {
    return std::pair<A, B>(extract<A>(obj[0]), extract<B>(obj[1]));
  }

  void face_group_add_face(face_group &fg, object &my_ind_py, object &opp_ind_py, 
      const flux::face &face)
  {
    face_group::index_list my_ind, opp_ind;

    for (unsigned i = 0; i < unsigned(len(my_ind_py)); i++)
      my_ind.push_back(extract<unsigned>(my_ind_py[i]));
    for (unsigned i = 0; i < unsigned(len(my_ind_py)); i++)
      opp_ind.push_back(extract<unsigned>(opp_ind_py[i]));

    fg.add_face(my_ind, opp_ind, face);
  }

  void face_group_connect_faces(face_group &fg, object &cnx_list_py)
  {
    face_group::connection_list cnx_list;

    for (unsigned i = 0; i < unsigned(len(cnx_list_py)); i++)
      cnx_list.push_back(extract_pair<unsigned, unsigned>(cnx_list_py[i]));

    fg.connect_faces(cnx_list);
  }
}




void hedge_expose_face_operators()
{
  {
    typedef face_group cl;
    class_<cl>("FaceGroup")
      .def("__len__", &cl::size)
      .def("clear", &cl::clear)
      .def("add_face", face_group_add_face)
      .def("connect_faces", face_group_connect_faces)
      ;
  }

#define ARG_TYPES \
  const face_group &, const matrix &, flux::chained_flux,
  DEF_FOR_EACH_OP_TARGET(perform_both_fluxes_operator, ARG_TYPES);
  DEF_FOR_EACH_OP_TARGET(perform_local_flux_operator, ARG_TYPES);
  DEF_FOR_EACH_OP_TARGET(perform_neighbor_flux_operator, ARG_TYPES);
#undef ARG_TYPES
}

