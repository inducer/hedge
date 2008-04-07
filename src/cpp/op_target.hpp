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




#ifndef _KJFALKUT_HEDGE_OP_TARGET_HPP_INCLUDED
#define _KJFALKUT_HEDGE_OP_TARGET_HPP_INCLUDED




#include <stdexcept>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include "base.hpp"




namespace hedge {
  /** \page optarget Operator Targets
   *
   * Operator targets arise because we would like to treat matrix-bound and
   * matrix-free operators with one set of code. In addition, they make the
   * code independent of any particular sparse matrix representation.
   *
   * Two examples of valid operator targets are:
   *
   * - a sparse matrix container. This container records all the operations
   *   involved in building the operator. In order to "replay" the same
   *   operation later, one can then simply multiply a vector with that matrix.
   *
   * - a pair of vectors (\c in, \c out). In this case, the operations
   *   described to the target are immediately carried out on the vector \c in,
   *   the result being stored in \c out.
   *
   *   Naturally, one could also write a target to apply the same transformation
   *   to a large number of vectors at once.
   *
   * So abstractly, an operator target is a receiver for a description of a
   * linear operator. "Performer functions" are functions that describe linear
   * operators to operator targets. These performers are called with operator
   * targets as their arguments.
   *
   * Operator target classes are required to have the following members:
   *
   * - a working copy constructor.
   *
   * - \c value_type: The type of the underlying field of the linear operator.
   *   \c double and <tt> std::complex<double></tt> are likely the most common
   *   types here.
   *
   * - <tt>begin(unsigned height, unsigned width)</tt>: begin the description
   *   of the operator by announcing the input and output dimensions to the
   *   target. This may do such things as allocate matrix storage or allocate
   *
   * - <tt>add_coefficient(unsigned i, unsigned j, value_type coeff)</tt>: add
   *   the coefficient \c coeff at the zero-based indices \c i and \c j.
   *
   * - <tt>add_coefficients(unsigned i, unsigned j, unsigned h, unsigned w,
   *   const Container &coefficients)</tt>: Add a contiguous batch of \c h
   *   \f$\times\f$ \c w coefficients at the zero-based indices \c i and \c j. The
   *   Container will be accessed by <tt>value_type operator()(unsigned i,
   *   unsigned j)</tt> accesses, which it is assumed to support.
   *
   * - <tt>finalize()</tt>: 
   *
   * Note that these members functions should *not* be virtual--otherwise,
   * a dramatic runtime penalty is to be expected.
   *
   * Since all relevant types are known at compile time and all function
   * calls can thus be inlined, it is expected that this construction causes 
   * no appreciable runtime overhead.
   *
   * The operator target is expected to be a thin wrapper holding references
   * to the actual objects being acted upon, and thus is passed into the 
   * performer function by value, allowing it to be a temporary. This
   * enables notation such as
   *
   * \code
   *   my_sparse_matrix m;
   *   perform_something(my_sparse_matrix_target(m));
   * \endcode
   *
   * or
   *
   * \code
   *   my_vector in, out;
   *   perform_something(my_vector_target(out, in));
   * \endcode
   */




  class null_target {
    public:
      typedef py_vector::value_type scalar_type;

      void begin(unsigned height, unsigned width) const
      { }

      void finalize() const
      { }

      void add_coefficient(unsigned i, unsigned j, scalar_type coeff) const
      { }

      template <class Container>
      void add_coefficients(unsigned i_start, unsigned j_start, 
          const Container &submat)
      { }

      template <class Container>
      void add_scaled_coefficients(unsigned i_start, unsigned j_start, 
          scalar_type factor, const Container &submat)
      { }
  };




  class vector_target {
    public:
      typedef py_vector::value_type scalar_type;

      vector_target(const py_vector operand, py_vector result)
        : m_operand(operand), m_result(result)
      { }
      void begin(unsigned height, unsigned width) const
      {
        if (m_operand.size() != width)
          throw std::runtime_error(
              "operand size does not match target width");
        if (m_result.size() != height)
          throw std::runtime_error(
              "result size does not match target height");
      }

      void finalize() const
      { }

      void add_coefficient(unsigned i, unsigned j, scalar_type coeff)
      { m_result[i] += coeff*m_operand[j]; }

      template <class Container>
      void add_coefficients(unsigned i_start, unsigned j_start, 
          const Container &submat)
      {
        /*
        boost::numeric::ublas::vector_range<vector> target
          (m_result, boost::numeric::ublas::range(i_start, i_stop));
        axpy_prod(submat, subrange(m_operand, j_start, j_stop), target);
        */
        noalias(subrange(m_result, i_start, i_start + submat.size1())) +=
            prod(submat, subrange(m_operand, j_start, j_start + submat.size2()));
      }

      template <class Container>
      void add_scaled_coefficients(unsigned i_start, unsigned j_start, 
          scalar_type factor, const Container &submat)
      {
        noalias(subrange(m_result, i_start, i_start+submat.size1())) +=
          factor * prod(submat, subrange(m_operand, j_start, j_start+submat.size2()));
      }

      const py_vector m_operand;
      py_vector m_result;
  };




  template <class Mat>
  class matrix_target {
    public:
      typedef Mat matrix_type;
      typedef typename Mat::size_type index_type;
      typedef typename Mat::value_type scalar_type;

      matrix_target(matrix_type &matrix, index_type row_offset=0, index_type col_offset=0)
        : m_matrix(matrix), m_row_offset(row_offset), m_col_offset(col_offset)
      {
      }

      index_type row_offset() const
      { return m_row_offset; }

      index_type column_offset() const
      { return m_col_offset; }

      void begin(unsigned height, unsigned width) const
      { 
        if (height + m_row_offset > m_matrix.size1())
          throw std::range_error("matrix_target targets unavailable rows");
        if (width + m_col_offset > m_matrix.size2())
          throw std::range_error("matrix_target targets unavailable columns");
      }

      void finalize() const
      { }

      matrix_target rebased_target(index_type row_offset, index_type col_offset)
      {
        return matrix_target(m_matrix, 
            m_row_offset+row_offset, 
            m_col_offset+col_offset);
      }

      void add_coefficient(unsigned i, unsigned j, scalar_type coeff)
      { m_matrix.append_element(m_row_offset+i, m_col_offset+j, coeff); }

      template <class Container>
      void add_coefficients(unsigned i_start, unsigned j_start, 
          const Container &submat)
      { 
        typename Container::const_iterator1 
          first1 = submat.begin1(), last1 = submat.end1();

        i_start += m_row_offset;
        j_start += m_col_offset;

        while (first1 != last1)
        {
          typename Container::const_iterator2 
            first2 = first1.begin(), last2 = first1.end();

          while (first2 != last2)
          {
            m_matrix.append_element(
                i_start+first2.index1(),
                j_start+first2.index2(),
                *first2);
            ++first2;
          }

          ++first1;
        }
      }

      template <class Container>
      void add_scaled_coefficients(unsigned i_start, unsigned j_start, 
          scalar_type factor, const Container &submat)
      { 
        //subrange(m_matrix, i_start, i_stop, j_start, j_stop) += factor * submat;
        typename Container::const_iterator1 
          first1 = submat.begin1(), last1 = submat.end1();

        i_start += m_row_offset;
        j_start += m_col_offset;

        while (first1 != last1)
        {
          typename Container::const_iterator2 
            first2 = first1.begin(), last2 = first1.end();

          while (first2 != last2)
          {
            m_matrix.append_element(
                i_start+first2.index1(),
                j_start+first2.index2(),
                factor * *first2);
            ++first2;
          }
          ++first1;
        }
      }

    protected:
      matrix_type &m_matrix;
      const index_type m_row_offset, m_col_offset;
  };




  typedef matrix_target<boost::numeric::ublas::coordinate_matrix<double, 
          boost::numeric::ublas::column_major> > 
    coord_matrix_target;
}




#endif
