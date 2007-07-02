#include <boost/python.hpp>
#include "index_subset.hpp"
#include "wrap_helpers.hpp"
#include "op_target.hpp"




using namespace boost::python;
using namespace hedge;




void hedge_expose_index_subset()
{
  {
    typedef index_subset cl;
    class_<cl>("IndexSubset")
      .DEF_SIMPLE_METHOD(clear)
      .DEF_SIMPLE_METHOD(add_index)
      ;
  }

  DEF_FOR_EACH_OP_TARGET(perform_restriction, );
  DEF_FOR_EACH_OP_TARGET(perform_expansion, );
}
