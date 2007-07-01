#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/python.hpp>
#include "op_target.hpp"
#include "face_operators.hpp"
#include "wrap_helpers.hpp"




using namespace boost::python;
using namespace hedge;
namespace ublas = boost::numeric::ublas;




namespace
{
  void face_group_add_face(face_group &fg, object &my_ind, object &opp_ind, 
      const flux::face &face)
  {
    face_group::index_list my_ind_il, opp_ind_il;

    for (unsigned i = 0; i < len(my_ind); i++)
      my_ind_il.push_back(extract<unsigned>(my_ind[i]));
    for (unsigned i = 0; i < len(my_ind); i++)
      opp_ind_il.push_back(extract<unsigned>(opp_ind[i]));

    fg.add_face(my_ind_il, opp_ind_il, face);
  }
}




void hedge_expose_face_operators()
{
  {
    typedef face_group cl;
    class_<cl>("FaceGroup")
      .def("clear", &cl::clear)
      .def("add_face", face_group_add_face)
      ;
  }

#define FLUX_OPERATOR_TEMPLATE_ARGS matrix, flux::chained_flux
  DEF_FOR_EACH_OP_TARGET(perform_both_fluxes_operator, FLUX_OPERATOR_TEMPLATE_ARGS);
  DEF_FOR_EACH_OP_TARGET(perform_local_flux_operator, FLUX_OPERATOR_TEMPLATE_ARGS);
  DEF_FOR_EACH_OP_TARGET(perform_neighbor_flux_operator, FLUX_OPERATOR_TEMPLATE_ARGS);
}

