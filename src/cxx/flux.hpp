#ifndef _BADFJAH_HEDGE_FLUX_HPP_INCLUDED
#define _BADFJAH_HEDGE_FLUX_HPP_INCLUDED




#include "base.hpp"




namespace hedge { namespace flux {
  struct face
  {
    double h;
    double face_jacobian;
    unsigned order;
    hedge::vector normal;
  };




  class flux
  {
    public:
      virtual ~flux() { }
      virtual double neighbor_coeff(const face &local) const = 0;
      virtual double local_coeff(const face &local) const = 0;
  };





  class chained_flux : public flux
  {
    public:
      chained_flux(const flux &child)
        : m_child(child)
      { }
      double neighbor_coeff(const face &local) const
      { return m_child.neighbor_coeff(local); }
      double local_coeff(const face &local) const
      { return m_child.local_coeff(local); }

    private:
      const flux &m_child;
  };





#define HEDGE_FLUX_DECLARE_NORMAL(DIR, IDX) \
  class normal_##DIR : public flux \
  { \
    public: \
      double neighbor_coeff(const face &local) const \
      { return local.normal[IDX]; } \
      double local_coeff(const face &local) const \
      { return local.normal[IDX]; } \
  };

  HEDGE_FLUX_DECLARE_NORMAL(x,0);
  HEDGE_FLUX_DECLARE_NORMAL(y,1);
  HEDGE_FLUX_DECLARE_NORMAL(z,2);
#undef HEDGE_FLUX_DECLARE_NORMAL




  class zero : public flux
  {
    public:
      double neighbor_coeff(const face &local) const
      { return 0; }
      double local_coeff(const face &local) const
      { return 0; }
  };




  class half : public flux
  {
    public:
      double neighbor_coeff(const face &local) const
      { return 0.5; }
      double local_coeff(const face &local) const
      { return 0.5; }
  };




  class constant : public flux
  {
    public:
      constant(double value)
        : m_value(value)
      { }
      double neighbor_coeff(const face &local) const
      { return m_value; }
      double local_coeff(const face &local) const
      { return m_value; }
    protected:
      double m_value;
  };




  class local : public flux
  {
    public:
      double neighbor_coeff(const face &local) const
      { return 0; }
      double local_coeff(const face &local) const
      { return 1; }
  };




  class neighbor : public flux
  {
    public:
      double neighbor_coeff(const face &local) const
      { return 1; }
      double local_coeff(const face &local) const
      { return 0; }
  };




  class average : public flux
  {
    public:
      double neighbor_coeff(const face &local) const
      { return 0.5; }
      double local_coeff(const face &local) const
      { return 0.5; }
  };




  class trace_sign : public flux
  {
    public:
      double neighbor_coeff(const face &local) const
      { return 1; }
      double local_coeff(const face &local) const
      { return -1; }
  };




  class neg_trace_sign : public flux
  {
    public:
      double neighbor_coeff(const face &local) const
      { return -1; }
      double local_coeff(const face &local) const
      { return 1; }
  };




  class penalty_term : public flux
  {
    public:
      penalty_term(double coefficient, double power)
        : m_coefficient(coefficient), m_power(power)
      { }
      double neighbor_coeff(const face &local) const
      { return m_coefficient * pow(local.order*local.order/local.h, m_power); }
      double local_coeff(const face &local) const
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
      double neighbor_coeff(const face &local) const
      { 
        return m_operation(
            m_op1.neighbor_coeff(local),
            m_op2.neighbor_coeff(local)
            );
      }
      double local_coeff(const face &local) const
      { 
        return m_operation(
            m_op1.local_coeff(local),
            m_op2.local_coeff(local)
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
      double neighbor_coeff(const face &local) const
      { 
        return m_operation(m_op.neighbor_coeff(local));
      }
      double local_coeff(const face &local) const
      { 
        return m_operation(m_op.local_coeff(local));
      }

    protected:
      Operation m_operation;
      Operand m_op;
  };





  typedef binary_operator<
    std::multiplies<double>,
    neg_trace_sign,
    normal_x> jump_x;
  typedef binary_operator<
    std::multiplies<double>,
    neg_trace_sign,
    normal_y> jump_y;
  typedef binary_operator<
    std::multiplies<double>,
    neg_trace_sign,
    normal_z> jump_z;

    


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
      double neighbor_coeff(const face &local) const
      { 
        return m_operation(
            m_op1.neighbor_coeff(local),
            m_op2.neighbor_coeff(local)
            );
      }
      double local_coeff(const face &local) const
      { 
        return m_operation(
            m_op1.local_coeff(local),
            m_op2.local_coeff(local)
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
      double neighbor_coeff(const face &local) const
      { 
        return Operation()(
            m_op1.neighbor_coeff(local),
            m_op2
            );
      }
      double local_coeff(const face &local) const
      { 
        return m_operation(
            m_op1.local_coeff(local),
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
      double neighbor_coeff(const face &local) const
      { 
        return Operation()(m_op.neighbor_coeff(local));
      }
      double local_coeff(const face &local) const
      { 
        return m_operation(m_op.local_coeff(local));
      }

    protected:
      Operation m_operation;
      flux &m_op;
  };
} }




#endif
