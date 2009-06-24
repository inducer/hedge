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




#include <boost/utility.hpp>
#include "base.hpp"




namespace hedge { namespace fluxes {
  struct face
  {
    double h;
    double face_jacobian;
    element_number_t element_id;
    face_number_t face_id;
    unsigned order;
    bounded_vector<double, max_dims> normal;

    face()
      : h(0), face_jacobian(0), 
      element_id(INVALID_ELEMENT), face_id(INVALID_FACE),
      order(0)
    { }
  };

} }




#endif
