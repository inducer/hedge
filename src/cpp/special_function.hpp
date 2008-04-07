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




#ifndef _AFAHHHFAZ_HEDGE_SPECIAL_FUNCTION_HPP_INCLUDED
#define _AFAHHHFAZ_HEDGE_SPECIAL_FUNCTION_HPP_INCLUDED




#include <boost/tuple/tuple.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include "base.hpp"




namespace hedge {
  class jacobi_polynomial {
    public:
      jacobi_polynomial(double alpha, double beta, unsigned n)
        : m_alpha(alpha), m_beta(beta), m_n(n)
      {
        using boost::math::tgamma;

        m_gamma0 = pow(2, alpha+beta+1)/(alpha+beta+1)
          *tgamma(alpha+1)
          *tgamma(beta+1)
          /tgamma(alpha+beta+1);

        m_gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*m_gamma0;

        m_a.push_back(2/(2+alpha+beta)*sqrt((alpha+1)*(beta+1)/(alpha+beta+3)));
        m_b.push_back(0);

        for (unsigned i = 1; i < m_n; i++)
        {
          double h1 = 2*i+alpha+beta;
          m_a.push_back(
              2./(h1+2)*sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)*
                (i+1+beta)/(h1+1)/(h1+3)));
          m_b.push_back(- (alpha*alpha-beta*beta)/h1/(h1+2));
        }
      }
    
      double operator()(double x)
      {
        double llast = 1/sqrt(m_gamma0);
        if (m_n == 0) return llast;
        double last = ((m_alpha+m_beta+2)/2*x + (m_alpha-m_beta)/2)/sqrt(m_gamma1);
        if (m_n == 1) return last;

        unsigned i;
        for (i = 1; i < m_n-1; i++)
        {
          double current = 1/m_a[i]*(-m_a[i-1]*llast + (x-m_b[i])*last);
          llast = last;
          last = current;
        }
        double current = 1/m_a[i]*(-m_a[i-1]*llast + (x-m_b[i])*last);
        return current;
      }

    private:
      double m_alpha, m_beta;
      unsigned m_n;
      double m_gamma0, m_gamma1;
      std::vector<double> m_a, m_b;
  };



  
  class diff_jacobi_polynomial
  {
    public:
      diff_jacobi_polynomial(double alpha, double beta, unsigned n)
      {
        if (n == 0)
          m_factor = 0;
        else
        {
          m_jf = std::auto_ptr<jacobi_polynomial>(
              new jacobi_polynomial(alpha+1, beta+1, n-1));
          m_factor = sqrt(n*(n+alpha+beta+1));
        }
      }

      double operator()(double x)
      {
        if (m_jf.get())
          return m_factor * (*m_jf)(x);
        else
          return 0;
      }

    protected:
      std::auto_ptr<jacobi_polynomial> m_jf;
      double m_factor;
  };




  // element basis functions --------------------------------------------------
  class triangle_basis_function
  {
    public:
      triangle_basis_function(int i, unsigned j)
        : m_i(i), m_j(j), m_f(0, 0, i), m_g(2*i+1, 0, j)
      { }

      double operator()(const py_vector &x)
      {
        double r = x[0];
        double s = x[1];

        double a;
        if (1-s != 0)
          a = 2*(1+r)/(1-s)-1;
        else
          a = 1;

        return sqrt(2)*m_f(a)*m_g(s)*pow(1-s, m_i);
      }

    protected:
      unsigned m_i, m_j;
      jacobi_polynomial m_f, m_g;
  };




  class grad_triangle_basis_function
  {
    public:
      grad_triangle_basis_function(int i, int j)
        : m_i(i), m_j(j), m_f(0, 0, i), m_df(0, 0, i), m_g(2*i+1, 0, j), m_dg(2*i+1, 0, j)
      { }

      boost::tuple<double, double> operator()(const py_vector &x)
      {
        double r = x[0];
        double s = x[1];

        double a;
        if (1-s != 0)
          a = 2*(1+r)/(1-s)-1;
        else
          a = 1;

        double f_a = m_f(a);
        double g_s = m_g(s);
        double df_a = m_df(a);
        double dg_s = m_dg(s);

        double one_s = 1-s;
        int i = m_i;

        // see doc/hedge-notes.tm
        return boost::make_tuple(
            // df/dr
            2*sqrt(2) * g_s * pow(one_s, i-1) * df_a,
            // df/ds
            sqrt(2)*(
              f_a * pow(one_s, i) * dg_s
                +(2*r+2) * g_s * pow(one_s, i-2) * df_a
                -i * f_a * g_s * pow(one_s, i-1)
              ));
      }

    protected:
      int m_i, m_j;
      jacobi_polynomial m_f;
      diff_jacobi_polynomial m_df;
      jacobi_polynomial m_g;
      diff_jacobi_polynomial m_dg;
  };




  class tetrahedron_basis_function
  {
    public:
      tetrahedron_basis_function(int i, int j, int k)
        : m_i(i), m_j(j), m_k(k), m_f(0, 0, i), m_g(2*i+1, 0, j), m_h(2*i+2*j+2, 0, k)
      { }

      double operator()(const py_vector &x)
      {
        double r = x[0];
        double s = x[1];
        double t = x[2];

        double a;
        if ((s+t) != 0)
          a = -2*(1+r)/(s+t) - 1;
        else
            a = -1;

        double b;
        if ((1-t) != 0)
          b = 2*(1+s)/(1-t) - 1;
        else
          b = -1;

        double c = t;

        return sqrt(8) \
                *m_f(a) \
                *m_g(b) \
                *pow(1-b, m_i) \
                *m_h(c) \
                *pow(1-c, m_i+m_j);
      }

    protected:
      int m_i, m_j, m_k;
      jacobi_polynomial m_f, m_g, m_h;
  };



  class grad_tetrahedron_basis_function
  {
    public:
      grad_tetrahedron_basis_function(int i, int j, int k)
        : m_i(i), m_j(j), m_k(k),
        m_f(0, 0, i),
        m_df(0, 0, i),
        m_g(2*i+1, 0, j),
        m_dg(2*i+1, 0, j),
        m_h(2*i+2*j+2, 0, k),
        m_dh(2*i+2*j+2, 0, k)
      { }

      boost::tuple<double, double, double> operator()(const py_vector &x)
      {
        double r = x[0];
        double s = x[1];
        double t = x[2];

        double a;
        if ((s+t) != 0)
          a = -2*(1+r)/(s+t) - 1;
        else
            a = -1;

        double b;
        if ((1-t) != 0)
          b = 2*(1+s)/(1-t) - 1;
        else
          b = -1;

        double c = t;

        double fa = m_f(a);
        double gb = m_g(b);
        double hc = m_h(c);

        double dfa = m_df(a);
        double dgb = m_dg(b);
        double dhc = m_dh(c);

        int id = m_i;
        int jd = m_j;

        // shamelessly stolen from Hesthaven/Warburton's GradSimplex3DP

        double tmp, V3Dr, V3Ds, V3Dt;

        // r-derivative
        V3Dr = dfa*(gb*hc);
        if (id>0)    
          V3Dr = V3Dr*pow(0.5*(1-b), id-1);
        if (id+jd>0) 
          V3Dr = V3Dr*pow(0.5*(1-c), id+jd-1);

        // s-derivative 
        V3Ds = 0.5*(1+a)*V3Dr;
        tmp = dgb*pow(0.5*(1-b), id);
        if (id>0)
          tmp = tmp+(-0.5*id)*(gb*pow(0.5*(1-b), id-1));
        if (id+jd>0) 
          tmp = tmp*(pow(0.5*(1-c), id+jd-1));
        tmp = fa*(tmp*hc);
        V3Ds = V3Ds+tmp;

        // t-derivative 
        V3Dt = 0.5*(1+a)*V3Dr+0.5*(1+b)*tmp;
        tmp = dhc*pow(0.5*(1-c), id+jd);
        if (id+jd>0)
            tmp = tmp-0.5*(id+jd)*(hc*pow(0.5*(1-c), id+jd-1));
        tmp = fa*(gb*tmp);
        tmp = tmp*pow(0.5*(1-b), id);
        V3Dt = V3Dt+tmp;

        // normalize
        return boost::make_tuple(
         V3Dr*pow(2, 2*id+jd+1.5),
         V3Ds*pow(2, 2*id+jd+1.5),
         V3Dt*pow(2, 2*id+jd+1.5)
         );
      }


    protected:
      int m_i, m_j, m_k;
      jacobi_polynomial m_f;
      diff_jacobi_polynomial m_df;
      jacobi_polynomial m_g;
      diff_jacobi_polynomial m_dg;
      jacobi_polynomial m_h;
      diff_jacobi_polynomial m_dh;
  };
}



#endif
