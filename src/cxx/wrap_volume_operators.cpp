#include <boost/python.hpp>
#include "op_target.hpp"
#include "volume_operators.hpp"
#include "wrap_helpers.hpp"




using namespace boost::python;
using namespace hedge;
namespace ublas = boost::numeric::ublas;




namespace
{
  tuple element_ranges_getitem(const element_ranges &er, int i)
  {
    if (i < 0)
      i += er.size();
    if (i < 0 || i >= er.size())
      PYTHON_ERROR(IndexError, "element_ranges index out of bounds");

    const element_ranges::element_range &erng = er[i];
    return make_tuple(erng.first, erng.second);
  }
}




void hedge_expose_volume_operators()
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

#define VOLUME_OPERATORS_TEMPLATE_ARGS matrix,
  DEF_FOR_EACH_OP_TARGET(perform_elwise_operator, VOLUME_OPERATORS_TEMPLATE_ARGS);
  DEF_FOR_EACH_OP_TARGET(perform_elwise_scaled_operator, VOLUME_OPERATORS_TEMPLATE_ARGS);
}
