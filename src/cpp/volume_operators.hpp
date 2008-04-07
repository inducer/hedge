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
#include <boost/foreach.hpp>
#include <boost/numeric/bindings/blas/blas3.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/matrix_traits.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include "base.hpp"
#include "op_target.hpp"
#include "flux.hpp"




namespace hedge {
  typedef std::pair<unsigned, unsigned> element_range;




  struct nonuniform_element_ranges 
  {
    private:
      typedef std::vector<element_range> container;
      container m_container;

    public:
      nonuniform_element_ranges()
      { }

      const unsigned size() const
      { return m_container.size(); }
      void clear()
      { m_container.clear(); }
      void append_range(unsigned start, unsigned end)
      { m_container.push_back(std::make_pair(start, end)); }
      const element_range &operator[](unsigned i) const
      { return m_container[i]; }

      typedef container::const_iterator const_iterator;
      const const_iterator begin() const { return m_container.begin(); }
      const const_iterator end() const { return m_container.end(); }

      typedef const_iterator iterator;
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
  template <class SrcERanges, class DestERanges, class Mat, class OT>
  inline
  void perform_elwise_operator(
      const SrcERanges &src_ers, 
      const DestERanges &dest_ers,
      const Mat &matrix, OT target)
  {
    if (src_ers.size() != dest_ers.size())
      throw std::runtime_error("element ranges have different sizes");

    typename DestERanges::const_iterator dest_ers_it = dest_ers.begin();

    BOOST_FOREACH(const element_range src_er, src_ers)
    {
      const element_range dest_er = *dest_ers_it++;
      target.add_coefficients(dest_er.first, src_er.first, matrix);
    }
  }




  template <class SrcERanges, class DestERanges, class Mat, class OT>
  inline
  void perform_elwise_scaled_operator(
      const SrcERanges &src_ers, 
      const DestERanges &dest_ers,
      const py_vector &scale_factors, const Mat &matrix, OT target)
  {
    if (src_ers.size() != dest_ers.size())
      throw std::runtime_error("element ranges have different sizes");

    unsigned i = 0;
    BOOST_FOREACH(const element_range src_er, src_ers)
    {
      const element_range dest_er = dest_ers[i];
      target.add_scaled_coefficients(dest_er.first, src_er.first, 
          scale_factors[i++], matrix);
    }
  }




  template <class ERanges>
  inline
  void perform_elwise_scale(const ERanges &ers, 
      py_vector const &scale_factors, vector_target tgt)
  {
    unsigned i = 0;
    BOOST_FOREACH(const element_range er, ers)
    {
      noalias(subrange(tgt.m_result, er.first, er.second)) += 
        scale_factors[i++] * 
        subrange(tgt.m_operand, er.first, er.second);
    }
  }




  // fast specializations -----------------------------------------------------
#ifdef USE_BLAS
  template <class Mat>
  inline
  void perform_elwise_scaled_operator(
      const uniform_element_ranges &src_ers, 
      const uniform_element_ranges &dest_ers, 
      const py_vector &scale_factors, const Mat &matrix, vector_target target)
  {
    if (src_ers.size()*src_ers.el_size() != target.m_operand.size())
      throw std::runtime_error("operand is of wrong size");
    if (dest_ers.size()*dest_ers.el_size() != target.m_result.size())
      throw std::runtime_error("operand is of wrong size");

    unsigned i = 0;
    py_vector new_operand(target.m_operand.size());
    BOOST_FOREACH(const element_range r, src_ers)
    {
      noalias(subrange(new_operand, r.first, r.second)) = 
        scale_factors[i++] * subrange(target.m_operand, r.first, r.second);
    }

    perform_elwise_operator(src_ers, dest_ers, matrix, 
        vector_target(new_operand, target.m_result));
  }





  template <class Mat>
  inline
  void perform_elwise_operator(
      const uniform_element_ranges &src_ers, 
      const uniform_element_ranges &dest_ers, 
      const Mat &matrix, vector_target target)
  {
    if (src_ers.size() != dest_ers.size())
      throw std::runtime_error("element ranges have different sizes");
    if (matrix.size2() != src_ers.el_size())
      throw std::runtime_error("number of matrix columns != size of src element");
    if (matrix.size1() != dest_ers.el_size())
      throw std::runtime_error("number of matrix rows != size of dest element");
    if (src_ers.size()*src_ers.el_size() != target.m_operand.size())
      throw std::runtime_error("operand is of wrong size");
    if (dest_ers.size()*dest_ers.el_size() != target.m_result.size())
      throw std::runtime_error("operand is of wrong size");

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
        /*b*/ traits::vector_storage(target.m_operand) + src_ers.start(), 
        /*ldb*/ src_ers.el_size(),
        /*beta*/ 1,
        /*c*/ traits::vector_storage(target.m_result) + dest_ers.start(), 
        /*ldc*/ dest_ers.el_size()
        );
  }
#endif
}




#endif

