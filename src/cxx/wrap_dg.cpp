#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/python.hpp>
#include "op_target.hpp"
#include "primitives.hpp"
#include "wrap_helpers.hpp"




using namespace boost::python;
using namespace hedge;
namespace ublas = boost::numeric::ublas;




namespace
{
  template <class cl>
  void expose_op_target(class_<cl> &wrapper)
  {
    wrapper
      .def("begin", &cl::begin)
      .def("finalize", &cl::finalize)
      .def("add_coefficient", &cl::add_coefficient)
      .def("add_coefficients", 
          (void (cl::*)(unsigned, unsigned, unsigned, unsigned, const matrix &) const)
          &cl::add_coefficients)
      ;
  }

  tuple element_ranges_getitem(const element_ranges &eg, int i)
  {
    if (i < eg.m_first_element || i >= eg.m_first_element+eg.size())
      PYTHON_ERROR(IndexError, "element_ranges index out of bounds");

    const element_ranges::element_range &er = eg[i];
    return make_tuple(er.first, er.second);
  }

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




void hedge_expose_dg()
{
  {
    typedef element_ranges cl;
    class_<cl>("ElementRanges", init<unsigned>())
      .def("__len__", &cl::size)
      .def("clear", &cl::clear)
      .def("append_range", &cl::append_range)
      .def("__getitem__", element_ranges_getitem)
      ;
  }

  typedef matrix_target<ublas::coordinate_matrix<double> > my_matrix_target;

  def("perform_elwise_operator", perform_elwise_operator<matrix, vector_target>);
  def("perform_elwise_operator", perform_elwise_operator<matrix, my_matrix_target>);

  def("perform_elwise_scaled_operator", 
      perform_elwise_scaled_operator<matrix, vector_target>);
  def("perform_elwise_scaled_operator", 
      perform_elwise_scaled_operator<matrix, my_matrix_target>);

  {
    typedef vector_target cl;
    class_<cl> wrapper("VectorTarget", 
        init<const vector &, vector &>()
        [with_custodian_and_ward<1,2, with_custodian_and_ward<1,3> >()]
        );
    expose_op_target(wrapper);
  }

  {
    typedef my_matrix_target cl;
    class_<cl> wrapper("MatrixTarget", 
        init<my_matrix_target::matrix_type &>()
        [with_custodian_and_ward<1,2>()]
        );
    expose_op_target(wrapper);
  }

  {
    typedef face_group cl;
    class_<cl>("FaceGroup")
      .def("clear", &cl::clear)
      .def("add_face", face_group_add_face)
      ;
  }

  def("perform_interior_flux_operator", 
      perform_interior_flux_operator<matrix, flux::chained_flux, vector_target>);
  def("perform_interior_flux_operator", 
      perform_interior_flux_operator<matrix, flux::chained_flux, my_matrix_target>);
}
