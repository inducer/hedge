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




#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>




namespace hedge {
  typedef boost::numeric::ublas::vector<double> vector;
  typedef boost::numeric::ublas::matrix<double> matrix;

  class affine_map
  {
    public:
      matrix m_matrix;
      vector m_vector;
      double m_jacobian;

      affine_map(const matrix &mat, const vector &vec, const double &jac)
        : m_matrix(mat), m_vector(vec), m_jacobian(jac)
      { }

      vector operator()(const vector &op)
      {
        return prod(m_matrix, op) + m_vector;
      }
  };
}



#endif
