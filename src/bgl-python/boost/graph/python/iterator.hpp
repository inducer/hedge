// Copyright 2005 The Trustees of Indiana University.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  Authors: Douglas Gregor
//           Andrew Lumsdaine
#ifndef BOOST_PARALLEL_GRAPH_PYTHON_ITERATOR_HPP
#define BOOST_PARALLEL_GRAPH_PYTHON_ITERATOR_HPP

#include <string>
#include <boost/python.hpp>
#include <iterator>

namespace boost { namespace graph { namespace python {

namespace detail {
  template<typename T> bool type_already_registered()
  {
    using boost::python::objects::registered_class_object;
    using boost::python::type_id;
    
    return registered_class_object(type_id<T>()).get() != 0;
  }
} // end namespace detail

template<typename Iterator>
class simple_python_iterator
{
public:
  typedef typename std::iterator_traits<Iterator>::difference_type
  difference_type;

  simple_python_iterator(std::pair<Iterator, Iterator> p)
    : orig_first(p.first), first(p.first), last(p.second), n(-1) { }

  typename std::iterator_traits<Iterator>::value_type next() 
  { 
    using boost::python::objects::stop_iteration_error;

    if (first == last) stop_iteration_error();
    return *first++;
  }

  difference_type len()
  { 
    if (n == -1) n = std::distance(first, last);
    return n; 
  }

  static void declare(const char* name, const char* docstring = 0)
  {
    using boost::python::class_;
    using boost::python::no_init;
    using boost::python::objects::identity_function;
    if (!detail::type_already_registered<simple_python_iterator>())
      class_<simple_python_iterator<Iterator> >(name, docstring, no_init)
        .def("__iter__", identity_function())
        .def("__len__", &simple_python_iterator<Iterator>::len)
        .def("next", &simple_python_iterator<Iterator>::next)
      ;
  }

private:
  Iterator orig_first;
  Iterator first;
  Iterator last;
  difference_type n;
};

} } } // end namespace boost::graph::python

#endif // BOOST_PARALLEL_GRAPH_PYTHON_ITERATOR_HPP
