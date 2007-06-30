#ifndef _KJFALKUT_HEDGE_OP_TARGET_HPP_INCLUDED
#define _KJFALKUT_HEDGE_OP_TARGET_HPP_INCLUDED




#include <stdexcept>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
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

  class vector_target {
    public:
      typedef double value_type;

      vector_target(const vector &operand, vector &result)
        : m_operand(operand), m_result(result)
      { }
      void begin(unsigned height, unsigned width) const
      {
        if (m_operand.size() != width)
          throw std::runtime_error(
              "operand size does not match matrix width");
        if (m_result.size() != height)
          throw std::runtime_error(
              "result size does not match matrix height");
      }

      void finalize() const
      { }

      void add_coefficient(unsigned i, unsigned j, 
          const value_type &coeff) const
      { m_result[i] += coeff*m_operand[j]; }

      template <class Container>
      void add_coefficients(unsigned i_start, unsigned i_stop, 
          unsigned j_start, unsigned j_stop,
          const Container &submat) const
      {
        subrange(m_result, i_start, i_stop) += 
          prod(submat, subrange(m_operand, j_start, j_stop));
      }

      const vector &m_operand;
      vector &m_result;
  };

  template <class Mat>
  class matrix_target {
    public:
      typedef Mat matrix_type;
      typedef typename Mat::value_type value_type;

      matrix_target(matrix_type &matrix)
        : m_matrix(matrix)
      {
      }

      void begin(unsigned height, unsigned width) const
      { m_matrix.resize(height, width); }

      void finalize() const
      { m_matrix.sort(); }

      void add_coefficient(unsigned i, unsigned j, 
          const value_type &coeff) const
      { m_matrix.push_back(i, j, coeff); }

      template <class Container>
      void add_coefficients(unsigned i_start, unsigned i_stop, 
          unsigned j_start, unsigned j_stop,
          const Container &submat) const
      { subrange(m_matrix, i_start, i_stop, j_start, j_stop) += submat; }

    protected:
      matrix_type &m_matrix;
  };
}




#endif
