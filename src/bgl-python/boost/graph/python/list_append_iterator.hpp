// Copyright 2005 The Trustees of Indiana University.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  Authors: Douglas Gregor
//           Andrew Lumsdaine
#ifndef BOOST_GRAPH_PYTHON_LIST_APPEND_ITERATOR_HPP
#define BOOST_GRAPH_PYTHON_LIST_APPEND_ITERATOR_HPP

#include <boost/python.hpp>
#include <iterator>

namespace boost { namespace graph { namespace python {

class list_append_iterator 
  : public std::iterator<std::output_iterator_tag, void, void, void, void>
{
  struct append_proxy
  {
    append_proxy(list_append_iterator* self) : self(self) { }

    template<typename T>
    const T& operator=(const T& value)
    {
      self->values->append(value);
      return value;
    }
    
  private:
    list_append_iterator* self;
  };
  friend struct append_proxy;

public:
  list_append_iterator() : values() { }
  list_append_iterator(boost::python::list& values) : values(&values) { }
  
  append_proxy operator*() { return append_proxy(this); }
  
  list_append_iterator& operator++() { return *this; }
  list_append_iterator operator++(int) { return *this; }

protected:
  boost::python::list* values;
};

} } } // end namespace boost::graph::python


#endif // BOOST_GRAPH_PYTHON_LIST_APPEND_ITERATOR_HPP
