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
