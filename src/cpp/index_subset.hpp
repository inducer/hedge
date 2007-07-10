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




#ifndef _AFAYTHF_HEDGE_INDEX_GROUP_HPP_INCLUDED
#define _AFAYTHF_HEDGE_INDEX_GROUP_HPP_INCLUDED




#include <boost/foreach.hpp>
#include <vector>
#include <stdexcept>
#include "base.hpp"




namespace hedge {
  struct index_subset {
    std::vector<unsigned> m_full_indices;

    void clear()
    {
      m_full_indices.clear();
    }
    void add_index(unsigned i, unsigned full_idx)
    {
      if (i != m_full_indices.size())
        throw std::runtime_error("index_subset pair added out-of-order");
      m_full_indices.push_back(full_idx);
    }
  };

  template <class OT>
  inline
  void perform_restriction(const index_subset &iss, OT target)
  {
    unsigned i = 0;
    BOOST_FOREACH(unsigned full_index, iss.m_full_indices)
      target.add_coefficient(i++, full_index, 1);
  }

  template <class OT>
  inline
  void perform_expansion(const index_subset &iss, OT target)
  {
    unsigned i = 0;
    BOOST_FOREACH(unsigned full_index, iss.m_full_indices)
      target.add_coefficient(full_index, i++, 1);
  }
}




#endif
