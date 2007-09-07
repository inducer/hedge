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




#include <boost/python.hpp>




void hedge_expose_base();
void hedge_expose_fluxes();
void hedge_expose_op_target();
void hedge_expose_volume_operators();
void hedge_expose_polynomial();
void hedge_expose_index_map();
void hedge_expose_mpi();




BOOST_PYTHON_MODULE(_internal)
{
  hedge_expose_base();
  hedge_expose_fluxes();
  hedge_expose_op_target();
  hedge_expose_volume_operators();
  hedge_expose_polynomial();
  hedge_expose_index_map();
  hedge_expose_mpi();
}
