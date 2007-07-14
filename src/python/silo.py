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




class SiloFile(hedge._silo.DBFile):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, pathname, create=True, mode=None,
            fileinfo="Hedge visualization",
            target=DB_LOCAL, filetype=None):
        if create:
            if mode is None:
                mode = DB_NOCLOBBER
            if filetype is None:
                filetype = DB_PDB
            hedge._silo.DBFile.__init__(self, pathname, mode, target,
                    fileinfo, filetype)
        else:
            if mode is None:
                mode = DB_APPEND
            if filetype is None:
                filetype = DB_UNKNOWN
            hedge._silo.DBFile.__init__(self, pathname, mode, filetype)

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

    def put_pointmesh(self, mname, ndims, coords, optlist={}):
        hedge._silo.DBFile.put_pointmesh(self, mname, ndims, coords,
                _convert_optlist(optlist))

    def put_pointvar1(self, vname, mname, var, optlist={}):
        hedge._silo.DBFile.put_pointvar1(self, vname, mname, var,
                _convert_optlist(optlist))

    def put_pointvar(self, vname, mname, vars, optlist={}):
        hedge._silo.DBFile.put_pointvar(self, vname, mname, vars,
                _convert_optlist(optlist))

