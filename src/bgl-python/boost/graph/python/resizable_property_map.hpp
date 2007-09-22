// Copyright 2005 The Trustees of Indiana University.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  Authors: Douglas Gregor
//           Andrew Lumsdaine

#ifndef BOOST_GRAPH_PYTHON_RESIZABLE_PROPERTY_MAP_HPP
#define BOOST_GRAPH_PYTHON_RESIZABLE_PROPERTY_MAP_HPP

#include <vector>
#include <boost/vector_property_map.hpp>

namespace boost { namespace graph { namespace python {

class resizable_property_map : boost::noncopyable
{
 public:
  /* Invoked when a new value has been added to the end of the list of
     keys. The property map may resize it's internal data structure to
     accomodate this. Returns true so long as this property map is
     still "relevant" and should be kept. */ 
  virtual bool added_key() = 0;

  /* The key with the given index is being removed, and the value with
     the last index will replace it. Returns true so long as this
     property map is still "relevant" and should be kept. */
  virtual bool removed_key(std::size_t index) = 0;

  /* All of the keys in the graph have been shuffled. This vector maps
     from the old indices to the new indices. Returns true so long as
     this property map is still "relevant" and should be kept. */
  virtual bool shuffled_keys(const std::vector<std::size_t>& new_indices) = 0;

  virtual ~resizable_property_map() {}

 protected:
  resizable_property_map() {}
};

template<typename T, typename IndexMap>
class resizable_vector_property_map : public resizable_property_map
{
 public:
  typedef vector_property_map<T, IndexMap> property_map_type;

  resizable_vector_property_map(const property_map_type& pmap) : pmap(pmap) { }
  
  /* Invoked when a new value has been added to the end of the list of
     keys. The property map may resize it's internal data structure to
     accomodate this. */ 
  virtual bool added_key()
  {
    if (pmap.get_store().unique()) return false;
    pmap.get_store()->push_back(T());
    return true;
  }

  /* The key with the given index is being removed, and the value with
     the last index will replace it. */
  virtual bool removed_key(std::size_t index)
  {
    if (pmap.get_store().unique()) return false;
    pmap.get_store()->at(index) = pmap.get_store()->back();
    pmap.get_store()->pop_back();
    return true;
  }

  /* All of the keys in the graph have been shuffled. This vector maps
     from the old indices to the new indices. */
  virtual bool shuffled_keys(const std::vector<std::size_t>& new_indices)
  {
    if (pmap.get_store().unique()) return false;
    std::vector<T> new_storage(new_indices.size());
    for (std::size_t i = 0; i < new_indices.size(); ++i)
      new_storage[i] = pmap.get_store()->at(i);
    pmap.get_store()->swap(new_storage);
    return true;
  }

 protected:
  property_map_type pmap;
};

} } } // end namespace boost::graph::python

#endif // BOOST_GRAPH_PYTHON_RESIZABLE_PROPERTY_MAP_HPP
