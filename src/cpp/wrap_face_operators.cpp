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

    for (unsigned i = 0; i < len(my_ind_py); i++)
      my_ind.push_back(extract<unsigned>(my_ind_py[i]));
    for (unsigned i = 0; i < len(my_ind_py); i++)
      opp_ind.push_back(extract<unsigned>(opp_ind_py[i]));

    fg.add_face(my_ind, opp_ind, face);
  }

  void face_group_connect_faces(face_group &fg, object &cnx_list_py)
  {
    face_group::connection_list cnx_list;

    for (unsigned i = 0; i < len(cnx_list_py); i++)
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

#define FLUX_OPERATOR_TEMPLATE_ARGS matrix, flux::chained_flux,
  DEF_FOR_EACH_OP_TARGET(perform_both_fluxes_operator, FLUX_OPERATOR_TEMPLATE_ARGS);
  DEF_FOR_EACH_OP_TARGET(perform_local_flux_operator, FLUX_OPERATOR_TEMPLATE_ARGS);
  DEF_FOR_EACH_OP_TARGET(perform_neighbor_flux_operator, FLUX_OPERATOR_TEMPLATE_ARGS);
}

