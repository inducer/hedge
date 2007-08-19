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

#define ARG_TYPES const index_subset &, 
  DEF_FOR_EACH_OP_TARGET(perform_restriction, ARG_TYPES);
  DEF_FOR_EACH_OP_TARGET(perform_expansion, ARG_TYPES);
#undef ARG_TYPES
}
