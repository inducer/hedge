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




#ifndef _BADFJAH_HEDGE_FLUX_HPP_INCLUDED
#define _BADFJAH_HEDGE_FLUX_HPP_INCLUDED




#include <boost/tuple/tuple.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/clear.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/back_inserter.hpp>
#include "base.hpp"




namespace hedge { namespace flux {
  struct face
  {
    double h;
    double face_jacobian;
    unsigned element_id, face_id;
    unsigned order;
    hedge::vector normal;
  };




  class flux
  {
    public:
      virtual ~flux() { }
      virtual double local_coeff(const face &local) const = 0;
      virtual double neighbor_coeff(const face &local, const face *neighbor) const = 0;
  };





  class chained_flux : public flux
  {
    public:
      chained_flux(const flux &child)
        : m_child(child)
      { }
      double local_coeff(const face &local) const
      { return m_child.local_coeff(local); }
      double neighbor_coeff(const face &local, const face *neighbor) const
      { return m_child.neighbor_coeff(local, neighbor); }

    private:
      const flux &m_child;
  };





  class zero : public flux
  {
    public:
      double local_coeff(const face &local) const
      { return 0; }
      double neighbor_coeff(const face &local, const face *neighbor) const
      { return 0; }
  };




  class constant : public flux
  {
    public:
      constant(double local, double neighbor)
        : m_local(local), m_neighbor(neighbor)
      { }
      double local_coeff(const face &local) const
      { return m_local; }
      double neighbor_coeff(const face &local, const face *neighbor) const
      { return m_neighbor; }

      const double local_constant() const
      { return m_local; }
      const double neighbor_constant() const
      { return m_neighbor; }

    private:
      const double m_local, m_neighbor;
  };




  template<class Dir>
  class normal : public flux
  {
    public:
      typedef normal type;

      double local_coeff(const face &local) const
      { return local.normal[Dir::value]; }
      double neighbor_coeff(const face &local, const face *neighbor) const
      { return local.normal[Dir::value]; }

      static int direction()
      { return Dir::value; }

      static std::string name()
      { return "Normal" + boost::lexical_cast<std::string>(Dir::value) + "Flux"; }
  };




  template<class Dir>
  class constant_times_normal : public flux
  {
    public:
      constant_times_normal(double local, double neighbor)
        : m_local(local), m_neighbor(neighbor)
      { }

      double local_coeff(const face &local) const
      { return m_local * local.normal[Dir::value]; }
      double neighbor_coeff(const face &local, const face *neighbor) const
      { return m_neighbor * local.normal[Dir::value]; }

      static int direction()
      { return Dir::value; }

      static std::string name()
      { 
        return "ConstantTimesNormal" 
        + boost::lexical_cast<std::string>(Dir::value) 
        + "Flux"; 
      }

      const double local_constant() const
      { return m_local; }
      const double neighbor_constant() const
      { return m_neighbor; }

    private:
      const double m_local, m_neighbor;
  };




  template<class Dir1, class Dir2>
  class constant_times_2normals : public flux
  {
    public:
      constant_times_2normals(double local, double neighbor)
        : m_local(local), m_neighbor(neighbor)
      { }

      double local_coeff(const face &local) const
      { 
        return m_local 
          * local.normal[Dir1::value] 
          * local.normal[Dir2::value]; 
      }

      double neighbor_coeff(const face &local, const face *neighbor) const
      { 
        return m_neighbor 
        * local.normal[Dir1::value] 
        * local.normal[Dir2::value]; 
      }

      static boost::tuple<int,int> directions()
      { return boost::make_tuple(Dir1::value, Dir2::value); }

      static std::string name()
      { 
        return "ConstantTimes2Normal" 
        + boost::lexical_cast<std::string>(Dir1::value) 
        + boost::lexical_cast<std::string>(Dir2::value) 
        + "Flux"; 
      }

      const double local_constant() const
      { return m_local; }
      const double neighbor_constant() const
      { return m_neighbor; }

    private:
      const double m_local, m_neighbor;
  };




  class penalty_term : public flux
  {
    public:
      penalty_term(double coefficient, double power)
        : m_coefficient(coefficient), m_power(power)
      { }
      double local_coeff(const face &local) const
      { return m_coefficient * pow(local.order*local.order/local.h, m_power); }
      double neighbor_coeff(const face &local, const face *neighbor) const
      { return m_coefficient * pow(local.order*local.order/local.h, m_power); }
    protected:
      double m_coefficient, m_power;
  };




  /** A compile-time polymorphic face function consisting of a binary operation
   * on two underlying face functions.
   *
   * See also runtime_binary_operator for the compile-time-polymorphic version.
   */
  template<class Operation, class Operand1, class Operand2>
  class binary_operator : public flux
  {
    public:
      double local_coeff(const face &local) const
      { 
        return m_operation(
            m_op1.local_coeff(local),
            m_op2.local_coeff(local)
            );
      }
      double neighbor_coeff(const face &local, const face *neighbor) const
      { 
        return m_operation(
            m_op1.neighbor_coeff(local, neighbor),
            m_op2.neighbor_coeff(local, neighbor)
            );
      }

    protected:
      Operation m_operation;
      Operand1 m_op1;
      Operand2 m_op2;
  };





  /** A compile-time polymorphic face function consisting of a unary operation
   * on an underlying face functions.
   *
   * (This is "compile-time polymorphic" because the child nodes are known
   * at compile time, as opposed to polymorphism by virtual method, which would 
   * be run-time.)
   *
   * See also runtime_unary_operator for the compile-time-polymorphic version.
   */
  template<class Operation, class Operand>
  class unary_operator : public flux
  {
    public:
      unary_operator()
      { }
      unary_operator(const Operation &operation, const Operand &op)
        : m_operation(operation), m_op(op)
      { }
      double local_coeff(const face &local) const
      { 
        return m_operation(m_op.local_coeff(local));
      }
      double neighbor_coeff(const face &local, const face *neighbor) const
      { 
        return m_operation(m_op.neighbor_coeff(local, neighbor));
      }

    protected:
      Operation m_operation;
      Operand m_op;
  };





  /** A runtime-polymorphic face function consisting of a binary operation 
   * on two underlying face functions.
   *
   * See also binary_operator for the compile-time-polymorphic version.
   */
  template<class Operation>
  class runtime_binary_operator : public flux
  {
    public:
      runtime_binary_operator(
          flux &op1, 
          flux &op2)
        : m_op1(op1), m_op2(op2)
      { }
      double local_coeff(const face &local) const
      { 
        return m_operation(
            m_op1.local_coeff(local),
            m_op2.local_coeff(local)
            );
      }
      double neighbor_coeff(const face &local, const face *neighbor) const
      { 
        return m_operation(
            m_op1.neighbor_coeff(local, neighbor),
            m_op2.neighbor_coeff(local, neighbor)
            );
      }

    protected:
      Operation m_operation;
      flux &m_op1;
      flux &m_op2;
  };




  template<class Operation>
  class runtime_binary_operator_with_constant : public flux
  {
    public:
      runtime_binary_operator_with_constant(
          flux &op1, double op2)
        : m_op1(op1), m_op2(op2)
      { }
      double local_coeff(const face &local) const
      { 
        return m_operation(
            m_op1.local_coeff(local),
            m_op2
            );
      }
      double neighbor_coeff(const face &local, const face *neighbor) const
      { 
        return Operation()(
            m_op1.neighbor_coeff(local, neighbor),
            m_op2
            );
      }

    protected:
      Operation m_operation;
      flux &m_op1;
      double m_op2;
  };




  template<class Operation>
  class runtime_unary_operator : public flux
  {
    public:
      runtime_unary_operator(flux &op)
        : m_op(op)
      { }
      double local_coeff(const face &local) const
      { 
        return m_operation(m_op.local_coeff(local));
      }
      double neighbor_coeff(const face &local, const face *neighbor) const
      { 
        return Operation()(m_op.neighbor_coeff(local, neighbor));
      }

    protected:
      Operation m_operation;
      flux &m_op;
  };




  typedef boost::mpl::vector_c<int, 0, 1, 2> dim_list;

  typedef boost::mpl::transform<dim_list, 
          normal<boost::mpl::_1> >::type 
            normal_fluxes;

  typedef boost::mpl::transform<dim_list, 
          constant_times_normal<boost::mpl::_1> >::type
            constant_times_normal_fluxes;

  typedef boost::mpl::vector<
    // FIXME yugly and redundant. pending reply from boost users list
      constant_times_2normals<boost::mpl::int_<0>, boost::mpl::int_<0> >,
      constant_times_2normals<boost::mpl::int_<0>, boost::mpl::int_<1> >,
      constant_times_2normals<boost::mpl::int_<0>, boost::mpl::int_<2> >,

      constant_times_2normals<boost::mpl::int_<1>, boost::mpl::int_<0> >,
      constant_times_2normals<boost::mpl::int_<1>, boost::mpl::int_<1> >,
      constant_times_2normals<boost::mpl::int_<1>, boost::mpl::int_<2> >,

      constant_times_2normals<boost::mpl::int_<2>, boost::mpl::int_<0> >,
      constant_times_2normals<boost::mpl::int_<2>, boost::mpl::int_<1> >,
      constant_times_2normals<boost::mpl::int_<2>, boost::mpl::int_<2> >
      >
        constant_times_2normal_fluxes;

} }




#endif
