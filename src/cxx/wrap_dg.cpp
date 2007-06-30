#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/python.hpp>
#include "op_target.hpp"
#include "elementgroup.hpp"




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
}




void hedge_expose_dg()
{
  {
    typedef element_group cl;
    class_<cl>("ElementGroup")
      .def("size", &cl::size)
      .def("clear", &cl::clear)
      .def("add_range", &cl::add_range)
      ;
  }
  typedef matrix_target<ublas::coordinate_matrix<double> > my_matrix_target;

  def("apply_elwise_matrix", apply_elwise_matrix<vector_target, matrix>);
  def("apply_elwise_matrix", apply_elwise_matrix<my_matrix_target, matrix>);

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
}
