# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




import hedge._silo




for name, value in hedge._silo.symbols().iteritems():
    globals()[name] = value




def _convert_optlist(ol_dict):
    optcount = len(ol_dict) + 1
    ol = hedge._silo.DBOptlist(optcount, optcount * 150)

    for key, value in ol_dict.iteritems():
        if isinstance(value, int):
            ol.add_int_option(key, value)
        else:
            ol.add_option(key, value)

    return ol




class DBFile(hedge._silo.DBFile):
    def put_ucdmesh(self, name, ndims, coordnames, coords, 
            nzones, zonel_name, facel_name,
            optlist={}):
        hedge._silo.DBFile.put_ucdmesh(self, name, ndims, coordnames, coords, 
            nzones, zonel_name, facel_name, _convert_optlist(optlist))

    def put_ucdvar1(self, vname, mname, vec, centering, optlist={}):
        hedge._silo.DBFile.put_ucdvar1(self, vname, mname, vec, centering, 
                _convert_optlist(optlist))

    def put_ucdvar(self, vname, mname, varnames, vars, 
            centering, optlist={}):
        hedge._silo.DBFile.put_ucdvar(self, vname, mname, varnames, vars, centering, 
                _convert_optlist(optlist))

    def put_defvars(self, vname, vars):
        """Add an defined variable ("expression") to this database.

        The `vars' argument consists of a list of tuples of type
          (name, definition)
        or
          (name, definition, DB_VARTYPE_SCALAR | DB_VARTYPE_VECTOR).
        or even
          (name, definition, DB_VARTYPE_XXX, {options}).
        If the type is not specified, scalar is assumed.
        """
        
        hedge._silo.DBFile.put_defvars(self, vname, vars)
