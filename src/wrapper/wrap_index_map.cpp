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
#include <boost/python/stl_iterator.hpp>
#include "index_map.hpp"
#include "wrap_helpers.hpp"
#include "op_target.hpp"




using namespace boost::python;
using namespace hedge;




namespace
{
  from_index_map *make_from_index_map(
      unsigned from_length, unsigned to_length, object iterable)
  {
    std::auto_ptr<from_index_map> fim(
        new from_index_map(from_length, to_length));
    copy(
        stl_input_iterator<unsigned>(iterable), 
        stl_input_iterator<unsigned>(),
        back_inserter(fim->m_map));
    return fim.release();
  }



  unsigned from_index_map_len(const from_index_map &fim)
  { return fim.m_map.size(); }
  unsigned from_index_map_getitem(const from_index_map &fim, unsigned i)
  { return fim.m_map.at(i); }
}





void hedge_expose_index_map()
{
  {
    typedef from_index_map cl;
    class_<cl>("IndexMap", init<unsigned, unsigned>())
      .def("__init__", make_constructor(make_from_index_map))
      .def_readonly("from_length", &cl::m_from_length)
      .def_readonly("to_length", &cl::m_to_length)
      .def("__len__", from_index_map_len)
      .def("__getitem__", from_index_map_getitem)
      ;
  }

#define ARG_TYPES const from_index_map &, 
  DEF_FOR_EACH_OP_TARGET(perform_index_map, ARG_TYPES);
  DEF_FOR_EACH_OP_TARGET(perform_inverse_index_map, ARG_TYPES);
#undef ARG_TYPES
}
