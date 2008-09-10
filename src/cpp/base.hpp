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
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
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

  static const unsigned bounded_max_dims = 3;
  typedef boost::numeric::ublas::bounded_vector<double, bounded_max_dims> bounded_vector;
  typedef boost::numeric::ublas::bounded_vector<npy_int32, bounded_max_dims> bounded_int_vector;




  // basic linear algebra -----------------------------------------------------
  /* Matrix inversion 
   * Modified from original by Fredrik Orderud. 
   * Retrieved from Effective Ublas Wiki 2008-09-9.
   */
  template<class MatrixT>
  inline MatrixT invert_matrix(const MatrixT &input) 
  {
    using namespace boost::numeric::ublas;

    if (input.size1() != input.size2())
      throw std::runtime_error("det requires square matrix");

    typedef permutation_matrix<std::size_t> pmatrix;
    pmatrix pm(input.size1());

    // lu_factorize is in-place
    matrix<typename MatrixT::value_type> a(input);
    if (lu_factorize(a, pm))
      throw std::runtime_error("lu decomposition failed");

    MatrixT inverse(identity_matrix<typename MatrixT::value_type>(a.size1()));
    lu_substitute(a, pm, inverse);
    return inverse;
  }


  /*
  Compute sign of permutation `p` by counting the number of
  interchanges required to change the given permutation into the
  identity one.

  Algorithm from http://people.scs.fsu.edu/~burkardt/math2071/perm_sign.m
  */
  template <class Container>
  int permutation_sign(Container &p)
  {
    int n = p.size();
    int s = +1;

    for (int i = 0; i < n; i++)
    {
      // J is the current position of item I.
      int j = i;

      while (p[j] != i)
        j++;

      // Unless the item is already in the correct place, restore it.
      if (j != i)
      {
        std::swap(p[i], p[j]);
        s = -s;
      }
    }
    return s;
  }






  template<class MatrixT>
  inline typename MatrixT::value_type det(const MatrixT &input) 
  {
    using namespace boost::numeric::ublas;

    if (input.size1() != input.size2())
      throw std::runtime_error("det requires square matrix");

    const std::size_t n = input.size1();

    typedef permutation_matrix<std::size_t> pmatrix;
    pmatrix pm(n);

    typedef typename MatrixT::value_type value_type;
    typedef typename MatrixT::size_type size_type;

    // lu_factorize is in-place
    matrix<value_type> a(input);
    if (lu_factorize(a, pm))
      throw std::runtime_error("lu decomposition failed");

    vector<std::size_t> permut(n);
    for (std::size_t i = 0; i < n; i++) 
      permut[i] = i;
    for (std::size_t i = 0; i < n; i++) 
      std::swap(permut[i], permut[pm[i]]);
          
    value_type result = permutation_sign(permut);
    for (size_type i = 0; i < n; ++i)
      result *= a(i, i);
    return result;
  }




  class affine_map
  {
    private:
      hedge::py_matrix m_matrix;
      hedge::py_vector m_vector;
      mutable bool m_have_jacobian;
      mutable double m_jacobian;

    public:
      affine_map()
      { }

      affine_map(
          const hedge::py_matrix &mat, 
          const hedge::py_vector &vec)
        : m_matrix(mat), m_vector(vec), m_have_jacobian(false)
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

      const affine_map inverted() const
      {
        py_matrix inv = invert_matrix(m_matrix);
        return affine_map(inv, -prod(inv, m_vector));
      }

      double jacobian() const
      {
        if (m_have_jacobian)
          return m_jacobian;
        else
        {
          m_have_jacobian = true;
          m_jacobian = det(m_matrix);
          return m_jacobian;
        }
      }
  };




  // cross product 
  template <class VecType>
  inline typename VecType::value_type entry_or_zero(const VecType &v, unsigned i)
  {
    if (i >= v.size())
      return 0;
    else
      return v[i];
  }




  template <class T>
  inline T entry_or_zero(const T *v, int i)
  {
    return v[i];
  }




  template <class VecType1, class VecType2>
  inline
  const VecType1 cross(
      const VecType1 &a, 
      const VecType2 &b)
  {
    VecType1 result(3);
    result[0] = entry_or_zero(a,1)*entry_or_zero(b,2) - entry_or_zero(a,2)*entry_or_zero(b,1);
    result[1] = entry_or_zero(a,2)*entry_or_zero(b,0) - entry_or_zero(a,0)*entry_or_zero(b,2);
    result[2] = entry_or_zero(a,0)*entry_or_zero(b,1) - entry_or_zero(a,1)*entry_or_zero(b,0);
    return result;
  }




}



#endif
