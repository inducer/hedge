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




#ifndef _BADFJAH_HEDGE_BASE_HPP_INCLUDED
#define _BADFJAH_HEDGE_BASE_HPP_INCLUDED




#include <vector>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>




namespace hedge {
  typedef unsigned element_number;
  typedef unsigned face_number;
  typedef unsigned vertex_number;
  typedef unsigned node_index;

  typedef std::vector<element_number> el_id_vector;
  typedef std::vector<vertex_number> vtx_id_vector;
  typedef std::pair<element_number, face_number> el_face;

  static const element_number INVALID_ELEMENT = UINT_MAX;
  static const vertex_number INVALID_VERTEX = UINT_MAX;

  typedef boost::numeric::ublas::vector<double> vector;
  typedef boost::numeric::ublas::matrix<double> matrix;

  class affine_map
  {
    private:
      hedge::matrix m_matrix;
      hedge::vector m_vector;

    public:
      affine_map()
      { }

      affine_map(
          const hedge::matrix &mat, 
          const hedge::vector &vec)
        : m_matrix(mat), m_vector(vec)
      { }

      const hedge::vector operator()(const hedge::vector &op) const
      {
        return prod(m_matrix, op) + m_vector;
      }

      const hedge::vector &vector() const
      { return m_vector; }

      const hedge::matrix &matrix() const
      { return m_matrix; }
  };
}



#endif
