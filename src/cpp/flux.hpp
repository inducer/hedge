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




#include <boost/typeof/typeof.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/utility.hpp>
#include "base.hpp"




namespace hedge { namespace fluxes {
  struct face
  {
    double h;
    double face_jacobian;
    element_number element_id;
    face_number face_id;
    unsigned order;
    hedge::py_vector normal;
  };




  template<class Operation, class Operand>
  class unary_operator;
  template<class Operation, class Operand1, class Operand2>
  class binary_operator;
  class constant;




  template <class Derived>
  struct flux_operators
  {
    template <class Flux2>
    binary_operator<std::plus<double>, Derived, Flux2> 
      operator+(const flux_operators<Flux2> &f2) const
    {
      return binary_operator<std::plus<double>, Derived, Flux2>
        (static_cast<const Derived &>(*this), 
         static_cast<const Flux2 &>(f2));
    }

    template <class Flux2>
    binary_operator<std::minus<double>, Derived, Flux2> 
      operator-(const flux_operators<Flux2> &f2) const
    {
      return binary_operator<std::minus<double>, Derived, Flux2>
        (static_cast<const Derived &>(*this), 
         static_cast<const Flux2 &>(f2));
    }

    unary_operator<std::negate<double>, Derived> 
      operator-() const
    {
      return unary_operator<std::negate<double>, Derived>
        (static_cast<const Derived &>(*this));
    }

    binary_operator<std::minus<double>, Derived, constant> 
      operator*(const double d) const
    {
      return binary_operator<std::minus<double>, Derived, constant>
        (static_cast<const Derived &>(*this), constant(d));
    }

    template <class Flux2>
    binary_operator<std::multiplies<double>, Derived, Flux2> 
      operator*(const flux_operators<Flux2> &f2) const
    {
      return binary_operator<std::multiplies<double>, Derived, Flux2>
        (static_cast<const Derived &>(*this), 
         static_cast<const Flux2 &>(f2));
    }
  };




  class flux
  {
    public:
      virtual ~flux() { }
      virtual double operator()(const face &local, const face *neighbor) const = 0;
  };





  class chained_flux : public flux, public flux_operators<chained_flux>
  {
    public:
      chained_flux(const flux &child)
        : m_child(child)
      { }
      double operator()(const face &local, const face *neighbor) const
      { return m_child(local, neighbor); }

      const flux &child() const
      { return m_child; }

    private:
      const flux & m_child;
  };





  class constant : public flux, public flux_operators<constant>
  {
    public:
      constant(double value)
        : m_value(value)
      { }
      double operator()(const face &local, const face *neighbor) const
      { return m_value; }

      const double value() const
      { return m_value; }

    private:
      const double m_value;
  };

  inline constant operator+(const constant &self, const constant &other)
  { 
    return constant(
        self.value() + other.value());
  }
  inline constant operator-(const constant &self, const constant &other)
  { 
    return constant(self.value() - other.value());
  }
  inline constant operator-(const constant &self)
  { return constant(-self.value()); }
  inline constant operator*(const constant &self, const double c)
  { return constant(self.value() * c); }




  class normal : public flux, public flux_operators<normal>
  {
    public:
      normal(int axis)
        : m_axis(axis)
      { }

      double operator()(const face &local, const face *neighbor) const
      { return local.normal[m_axis]; }

      const int axis() const
      { return m_axis; }

    private:
      const int m_axis;
  };




  class penalty_term : public flux, public flux_operators<penalty_term>
  {
    public:
      penalty_term(double power)
        : m_power(power)
      { }
      double operator()(const face &local, const face *neighbor) const
      { 
        return pow(local.order*local.order/local.h, m_power); 
      }

      const double power() const
      { return m_power; }

    private:
      const double m_power;
  };




  template<class Criterion, class ThenPart, class ElsePart>
  class if_positive : 
    public flux, 
    public flux_operators<if_positive<Criterion, ThenPart, ElsePart> >
  {
    public:
      if_positive()
      { }

      if_positive(
          const ThenPart &thp,
          const ElsePart &elp)
        : m_then_part(thp), m_else_part(elp)
      { }

      if_positive(const Criterion &crit,
          const ThenPart &thp,
          const ElsePart &elp)
        : m_criterion(crit), m_then_part(thp), m_else_part(elp)
      { }

      double operator()(const face &local, const face *neighbor) const
      { 
        if (m_criterion(local, neighbor) > 0)
          return m_then_part(local, neighbor);
        else
          return m_else_part(local, neighbor);
      }

      const Criterion &criterion() const
      { return m_criterion; }

      const ThenPart &then_part() const
      { return m_then_part; }

      const ElsePart &else_part() const
      { return m_else_part; }

    protected:
      const Criterion m_criterion;
      const ThenPart m_then_part;
      const ElsePart m_else_part;
  };




  template<class Operation, class Operand1, class Operand2>
  class binary_operator : 
    public flux, 
    public flux_operators<binary_operator<Operation, Operand1, Operand2> >
  {
    public:
      binary_operator()
      { }

      binary_operator(const Operand1 &op1, const Operand2 &op2)
        : m_operation(), m_op1(op1), m_op2(op2)
      { }

      binary_operator(const Operation &operation, const Operand1 &op1, const Operand2 &op2)
        : m_operation(operation), m_op1(op1), m_op2(op2)
      { }

      double operator()(const face &local, const face *neighbor) const
      { 
        return m_operation(
            m_op1(local, neighbor),
            m_op2(local, neighbor)
            );
      }

      const Operand1 &operand1() const
      { return m_op1; }

      const Operand2 &operand2() const
      { return m_op2; }

    protected:
      const Operation m_operation;
      const Operand1 m_op1;
      const Operand2 m_op2;
  };




  template<class Operation, class Operand>
  class unary_operator : 
    public flux, 
    public flux_operators<unary_operator<Operation, Operand> >
  {
    public:
      unary_operator()
      { }
      unary_operator(const Operand &op)
        : m_operation(), m_op(op)
      { }
      unary_operator(const Operation &operation, const Operand &op)
        : m_operation(operation), m_op(op)
      { }
      double operator()(const face &local, const face *neighbor) const
      { 
        return m_operation(m_op(local, neighbor));
      }

      const Operand &operand() const
      { return m_op; }

    protected:
      const Operation m_operation;
      const Operand m_op;
  };
} }




#include BOOST_TYPEOF_INCREMENT_REGISTRATION_GROUP()
 
BOOST_TYPEOF_REGISTER_TYPE(hedge::fluxes::flux);
BOOST_TYPEOF_REGISTER_TYPE(hedge::fluxes::constant);
BOOST_TYPEOF_REGISTER_TYPE(hedge::fluxes::normal);
BOOST_TYPEOF_REGISTER_TYPE(hedge::fluxes::penalty_term);
BOOST_TYPEOF_REGISTER_TEMPLATE(hedge::fluxes::binary_operator, 3);
BOOST_TYPEOF_REGISTER_TEMPLATE(hedge::fluxes::unary_operator, 2);




#endif
