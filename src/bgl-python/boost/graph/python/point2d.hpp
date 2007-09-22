// Copyright 2005 The Trustees of Indiana University.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  Authors: Douglas Gregor
//           Andrew Lumsdaine
#ifndef BOOST_GRAPH_PYTHON_POINT2D_HPP
#define BOOST_GRAPH_PYTHON_POINT2D_HPP

#include <iostream>
#include <boost/graph/point_traits.hpp>

namespace boost { namespace graph { namespace python {
  class point2d
  {
  public:
    point2d() { coordinates[0] = coordinates[1] = 0.0f; }

    point2d(float x, float y) 
    {
      coordinates[0] = x;
      coordinates[1] = y;
    }

    float&       operator[](std::size_t i)       { return coordinates[i]; }
    const float& operator[](std::size_t i) const { return coordinates[i]; }

  private:
    float coordinates[2];
  };

  inline std::ostream& operator<<(std::ostream& out, point2d p)
  { return out << p[0] << p[1]; }

  inline std::istream& operator>>(std::istream& in, point2d& p)
  { return in >> p[0] >> p[1]; }

  inline bool operator==(const point2d& p1, const point2d& p2)
  {
    return p1[0] == p2[0] && p1[1] == p2[1];
  }

  inline bool operator!=(const point2d& p1, const point2d& p2)
  {
    return p1[0] != p2[0] || p1[1] != p2[1];
  }
} // end namespace python

template<>
struct point_traits<boost::graph::python::point2d>
{
  typedef float component_type;

  static std::size_t dimensions(const boost::graph::python::point2d&)
  { return 2; }
};

} } // end namespace boost::graph

#endif // BOOST_GRAPH_PYTHON_POINT2D_HPP
