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




#ifndef _AHFYHAT_HEDGE_VOLUME_OPERATORS_HPP_INCLUDED
#define _AHFYHAT_HEDGE_VOLUME_OPERATORS_HPP_INCLUDED




#include <vector>
#include <utility>
#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/numeric/bindings/blas/blas3.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/matrix_traits.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include "base.hpp"




namespace hedge {
  typedef std::pair<unsigned, unsigned> element_range;




  struct nonuniform_element_ranges 
  {
    private:
      typedef std::vector<node_number_t> container;
      container m_el_starts;
      unsigned m_el_size;

    public:
      nonuniform_element_ranges(unsigned el_size)
        : m_el_size(el_size)
      { }

      const unsigned size() const
      { return m_el_starts.size(); }
      unsigned el_size() const
      { return m_el_size; }

      void clear()
      { m_el_starts.clear(); }
      void append_el(unsigned el_start)
      { m_el_starts.push_back(el_start); }
      const element_range operator[](unsigned i) const
      {
        node_number_t el_start = m_el_starts[i];
        return std::make_pair(el_start, el_start+m_el_size);
      }

      class to_element_range_map
      {
        private:
          typedef nonuniform_element_ranges parent;
          const parent          &m_parent;

        public:
          typedef node_number_t argument_type;
          typedef element_range result_type;

          to_element_range_map(parent const &prt)
            : m_parent(prt)
          { }

          result_type operator()(argument_type arg)
          {
            return std::make_pair(arg, arg+m_parent.m_el_size);
          }
      };

      // iterator functionality
      typedef boost::transform_iterator<
        to_element_range_map, container::const_iterator> const_iterator;

      const const_iterator begin() const 
      {
        return boost::make_transform_iterator(
          m_el_starts.begin(),
          to_element_range_map(*this));
      }

      const const_iterator end() const
      {
        return boost::make_transform_iterator(
          m_el_starts.end(),
          to_element_range_map(*this));
      }
  };




  struct uniform_element_ranges 
  {
    private:
      unsigned m_start, m_el_size, m_el_count;

    public:
      uniform_element_ranges(unsigned start, unsigned el_size, unsigned el_count)
        : m_start(start), m_el_size(el_size), m_el_count(el_count)
      { }

      const unsigned start() const
      { return m_start; }

      unsigned size() const
      { return m_el_count; }

      unsigned el_size() const
      { return m_el_size; }

      unsigned total_size() const
      { return m_el_size*m_el_count; }

      const element_range operator[](int i) const
      { 
        unsigned el_start = m_start + i*m_el_size;
        return std::make_pair(el_start, el_start+m_el_size);
      }

      // iterator functionality
      class const_iterator : public boost::iterator_facade<
                             const_iterator, 
                             const element_range, 
                             boost::random_access_traversal_tag,
                             const element_range>
      {
        private:
          typedef uniform_element_ranges parent;

          const parent          *m_parent;
          int                   m_index;

        public:
          const_iterator()
          { }
          explicit const_iterator(const parent &prnt, int index)
            : m_parent(&prnt), m_index(index)
          { }

        private:
          friend class boost::iterator_core_access;

          const reference dereference() const
          { return (*m_parent)[m_index]; }
          const bool equal(const const_iterator &z) const
          { return m_index == z.m_index; }

          void increment()
          { ++m_index; }
          void decrement()
          { --m_index; }
          void advance(difference_type n)
          { m_index += n; }

          const difference_type distance(const const_iterator &z) const
          { return z.m_index - m_index; }
      };

      typedef const_iterator iterator;

      const const_iterator begin() const { return const_iterator(*this, 0); }
      const const_iterator end() const { return const_iterator(*this, size()); }
  };




  // generic operations -------------------------------------------------------
  template <class ERanges, class Scalar>
  inline
  void perform_elwise_scale(const ERanges &ers, 
      numpy_vector<double> const &scale_factors, 
      numpy_vector<Scalar> const &operand,
      numpy_vector<Scalar> result)
  {
    unsigned i = 0;
    BOOST_FOREACH(const element_range er, ers)
    {
      noalias(subrange(result, er.first, er.second)) += 
        Scalar(scale_factors[i++]) * 
        subrange(operand, er.first, er.second);
    }
  }



  // non-BLAS versions --------------------------------------------------------
  template <class SrcERanges, class DestERanges, class Scalar>
  inline
  void perform_elwise_scaled_operator(
      const SrcERanges &src_ers, 
      const DestERanges &dest_ers,
      const numpy_vector<double> &scale_factors, 
      const matrix<Scalar> &mat,
      const numpy_vector<Scalar> &operand,
      numpy_vector<Scalar> result)
  {
    if (src_ers.size() != dest_ers.size())
      throw std::runtime_error("element ranges have different sizes");

    typedef typename numpy_matrix<Scalar>::size_type size_type;
    size_type h = mat.size1();
    size_type w = mat.size2();

    unsigned i = 0;
    BOOST_FOREACH(const element_range src_er, src_ers)
    {
      const element_range dest_er = dest_ers[i];

      noalias(subrange(result, dest_er.first, dest_er.first+h)) +=
        Scalar(scale_factors[i]) * prod(mat, subrange(operand, src_er.first, src_er.first+w));

      ++i;
    }
  }





  template <class SrcERanges, class DestERanges, class Scalar>
  inline
  void perform_elwise_operator(
      const SrcERanges &src_ers, 
      const DestERanges &dest_ers,
      const matrix<Scalar> &mat,
      const numpy_vector<Scalar> &operand,
      numpy_vector<Scalar> result)
  {
    if (src_ers.size() != dest_ers.size())
      throw std::runtime_error("element ranges have different sizes");

    typedef typename numpy_matrix<Scalar>::size_type size_type;
    size_type h = mat.size1();
    size_type w = mat.size2();

    unsigned i = 0;
    BOOST_FOREACH(const element_range src_er, src_ers)
    {
      const element_range dest_er = dest_ers[i];

      noalias(subrange(result, dest_er.first, dest_er.first+h)) +=
        prod(mat, subrange(operand, src_er.first, src_er.first+w));

      ++i;
    }
  }




  // BLAS versions ------------------------------------------------------------
#ifdef USE_BLAS
  template <class Scalar>
  inline
  void perform_elwise_scaled_operator_using_blas(
      const uniform_element_ranges &src_ers, 
      const uniform_element_ranges &dest_ers, 
      const numpy_vector<double> &scale_factors, 
      const numpy_matrix<Scalar> &matrix,
      numpy_vector<Scalar> const &operand,
      numpy_vector<Scalar> result)
  {
    if (src_ers.size()*src_ers.el_size() != operand.size())
      throw std::runtime_error("operand is of wrong size");
    if (dest_ers.size()*dest_ers.el_size() != result.size())
      throw std::runtime_error("result is of wrong size");

    unsigned i = 0;
    numpy_vector<Scalar> new_operand(operand.size());
    BOOST_FOREACH(const element_range r, src_ers)
    {
      noalias(subrange(new_operand, r.first, r.second)) = 
        Scalar(scale_factors[i++]) * subrange(operand, r.first, r.second);
    }

    perform_elwise_operator_using_blas(src_ers, dest_ers, matrix, new_operand, result);
  }





  template <class Scalar>
  inline
  void perform_elwise_operator_using_blas(
      const uniform_element_ranges &src_ers, 
      const uniform_element_ranges &dest_ers, 
      const numpy_matrix<Scalar> &matrix, 
      numpy_vector<Scalar> const &operand,
      numpy_vector<Scalar> result)
  {
    if (src_ers.size() != dest_ers.size())
      throw std::runtime_error("element ranges have different sizes");
    if (matrix.size2() != src_ers.el_size())
      throw std::runtime_error("number of matrix columns != size of src element");
    if (matrix.size1() != dest_ers.el_size())
      throw std::runtime_error("number of matrix rows != size of dest element");
    if (src_ers.total_size() != operand.size())
      throw std::runtime_error(
          boost::str(boost::format("operand is of wrong size %d (expected %d)")
          % operand.size()
          % src_ers.total_size()).c_str()
          );
    if (dest_ers.total_size() != result.size())
      throw std::runtime_error("result is of wrong size");

    using namespace boost::numeric::bindings;
    using blas::detail::gemm;

    gemm(
        'T', // "matrix" is row-major
        'N', // a contiguous array of vectors is column-major
        matrix.size1(),
        src_ers.size(),
        matrix.size2(),
        /*alpha*/ 1,
        /*a*/ boost::numeric::bindings::traits::matrix_storage(matrix.as_ublas()),
        /*lda*/ matrix.size2(),
        /*b*/ traits::vector_storage(operand) + src_ers.start(), 
        /*ldb*/ src_ers.el_size(),
        /*beta*/ 1,
        /*c*/ traits::vector_storage(result) + dest_ers.start(), 
        /*ldc*/ dest_ers.el_size()
        );
  }
#endif




  // other helpers ------------------------------------------------------------
  template <class ERanges, class Vector>
  inline
  void perform_elwise_max(const ERanges &ers, 
      const Vector &in, Vector out)
  {
    typename Vector::const_iterator in_it = in.begin();
    typename Vector::iterator out_it = out.begin();
    
    BOOST_FOREACH(const element_range er, ers)
    {
      std::fill(out_it+er.first, out_it+er.second,
          *std::max_element(in_it+er.first, in_it+er.second));
    }
  }
}




#endif

