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
#include <pyublas/numpy.hpp>




namespace hedge {
  typedef npy_uint element_number_t;
  typedef npy_uint face_number_t;
  typedef npy_uint vertex_number_t;
  typedef npy_uint node_number_t;

  typedef std::vector<element_number_t> el_id_vector;
  typedef std::vector<vertex_number_t> vtx_id_vector;
  typedef std::pair<element_number_t, face_number_t> el_face_t;

  static const element_number_t INVALID_ELEMENT = UINT_MAX;
  static const face_number_t INVALID_FACE = UINT_MAX;
  static const vertex_number_t INVALID_VERTEX = UINT_MAX;
  static const node_number_t INVALID_NODE = UINT_MAX;

  typedef pyublas::numpy_vector<double> py_vector;
  typedef pyublas::numpy_vector<npy_uint32> py_uint_vector;
  typedef pyublas::numpy_matrix<double> py_matrix;
  typedef pyublas::numpy_matrix<double, 
          boost::numeric::ublas::column_major> py_fortran_matrix;

  typedef boost::numeric::ublas::vector<double> dyn_vector;
  typedef boost::numeric::ublas::matrix<double> dyn_matrix;
  typedef boost::numeric::ublas::matrix<double, 
          boost::numeric::ublas::column_major> dyn_fortran_matrix;

  class affine_map
  {
    private:
      hedge::py_matrix m_matrix;
      hedge::py_vector m_vector;

    public:
      affine_map()
      { }

      affine_map(
          const hedge::py_matrix &mat, 
          const hedge::py_vector &vec)
        : m_matrix(mat), m_vector(vec)
      { }

      template <class VecType>
      const VecType operator()(const VecType &op) const
      {
        return prod(m_matrix, op) + m_vector;
      }

      template <class ResultType, class VecType>
      const ResultType apply(const VecType &op) const
      {
        return prod(m_matrix, op) + m_vector;
      }

      const hedge::py_vector &vector() const
      { return m_vector; }

      const hedge::py_matrix &matrix() const
      { return m_matrix; }
  };
}



#endif
