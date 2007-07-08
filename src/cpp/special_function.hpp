#ifndef _AFAHHHFAZ_HEDGE_SPECIAL_FUNCTION_HPP_INCLUDED
#define _AFAHHHFAZ_HEDGE_SPECIAL_FUNCTION_HPP_INCLUDED




#include <boost/math/tools/config.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <vector>
#include <cmath>
#include <iostream>




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
}



#endif
