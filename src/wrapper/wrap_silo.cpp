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




#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/scoped_array.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "wrap_helpers.hpp"
#include "base.hpp"

#ifdef USE_SILO
#include <silo.h>
#endif




using namespace boost::python;
namespace traits = boost::numeric::bindings::traits;




namespace 
{
  bool have_silo()
  {
#ifdef USE_SILO
    return true;
#else
    return false;
#endif
  }

#ifdef USE_SILO
  dict symbols()
  {
    dict result;
#define EXPORT_CONSTANT(NAME) \
    result[#NAME] = NAME

    /* Drivers */
    EXPORT_CONSTANT(DB_NETCDF);
    EXPORT_CONSTANT(DB_PDB);
    EXPORT_CONSTANT(DB_TAURUS);
    EXPORT_CONSTANT(DB_SDX);
    EXPORT_CONSTANT(DB_UNKNOWN);
    EXPORT_CONSTANT(DB_DEBUG);
    EXPORT_CONSTANT(DB_HDF5);
    EXPORT_CONSTANT(DB_EXODUS);

    /* Flags for DBCreate */
    EXPORT_CONSTANT(DB_CLOBBER);
    EXPORT_CONSTANT(DB_NOCLOBBER);

    /* Flags for DBOpen */
    EXPORT_CONSTANT(DB_READ);
    EXPORT_CONSTANT(DB_APPEND);

    /* Target machine for DBCreate */
    EXPORT_CONSTANT(DB_LOCAL);
    EXPORT_CONSTANT(DB_SUN3);
    EXPORT_CONSTANT(DB_SUN4);
    EXPORT_CONSTANT(DB_SGI);
    EXPORT_CONSTANT(DB_RS6000);
    EXPORT_CONSTANT(DB_CRAY);
    EXPORT_CONSTANT(DB_INTEL);

    /* Options */
    EXPORT_CONSTANT(DBOPT_ALIGN);
    EXPORT_CONSTANT(DBOPT_COORDSYS);
    EXPORT_CONSTANT(DBOPT_CYCLE);
    EXPORT_CONSTANT(DBOPT_FACETYPE);
    EXPORT_CONSTANT(DBOPT_HI_OFFSET);
    EXPORT_CONSTANT(DBOPT_LO_OFFSET);
    EXPORT_CONSTANT(DBOPT_LABEL);
    EXPORT_CONSTANT(DBOPT_XLABEL);
    EXPORT_CONSTANT(DBOPT_YLABEL);
    EXPORT_CONSTANT(DBOPT_ZLABEL);
    EXPORT_CONSTANT(DBOPT_MAJORORDER);
    EXPORT_CONSTANT(DBOPT_NSPACE);
    EXPORT_CONSTANT(DBOPT_ORIGIN);
    EXPORT_CONSTANT(DBOPT_PLANAR);
    EXPORT_CONSTANT(DBOPT_TIME);
    EXPORT_CONSTANT(DBOPT_UNITS);
    EXPORT_CONSTANT(DBOPT_XUNITS);
    EXPORT_CONSTANT(DBOPT_YUNITS);
    EXPORT_CONSTANT(DBOPT_ZUNITS);
    EXPORT_CONSTANT(DBOPT_DTIME);
    EXPORT_CONSTANT(DBOPT_USESPECMF);
    EXPORT_CONSTANT(DBOPT_XVARNAME);
    EXPORT_CONSTANT(DBOPT_YVARNAME);
    EXPORT_CONSTANT(DBOPT_ZVARNAME);
    EXPORT_CONSTANT(DBOPT_ASCII_LABEL);
    EXPORT_CONSTANT(DBOPT_MATNOS);
    EXPORT_CONSTANT(DBOPT_NMATNOS);
    EXPORT_CONSTANT(DBOPT_MATNAME);
    EXPORT_CONSTANT(DBOPT_NMAT);
    EXPORT_CONSTANT(DBOPT_NMATSPEC);
    EXPORT_CONSTANT(DBOPT_BASEINDEX);
    EXPORT_CONSTANT(DBOPT_ZONENUM);
    EXPORT_CONSTANT(DBOPT_NODENUM);
    EXPORT_CONSTANT(DBOPT_BLOCKORIGIN);
    EXPORT_CONSTANT(DBOPT_GROUPNUM);
    EXPORT_CONSTANT(DBOPT_GROUPORIGIN);
    EXPORT_CONSTANT(DBOPT_NGROUPS);
    EXPORT_CONSTANT(DBOPT_MATNAMES);
    EXPORT_CONSTANT(DBOPT_EXTENTS_SIZE);
    EXPORT_CONSTANT(DBOPT_EXTENTS);
    EXPORT_CONSTANT(DBOPT_MATCOUNTS);
    EXPORT_CONSTANT(DBOPT_MATLISTS);
    EXPORT_CONSTANT(DBOPT_MIXLENS);
    EXPORT_CONSTANT(DBOPT_ZONECOUNTS);
    EXPORT_CONSTANT(DBOPT_HAS_EXTERNAL_ZONES);
    EXPORT_CONSTANT(DBOPT_PHZONELIST);
    EXPORT_CONSTANT(DBOPT_MATCOLORS);
    EXPORT_CONSTANT(DBOPT_BNDNAMES);
    EXPORT_CONSTANT(DBOPT_REGNAMES);
    EXPORT_CONSTANT(DBOPT_ZONENAMES);
    EXPORT_CONSTANT(DBOPT_HIDE_FROM_GUI);

    /* Error trapping method */
    EXPORT_CONSTANT(DB_TOP);
    EXPORT_CONSTANT(DB_NONE);
    EXPORT_CONSTANT(DB_ALL);
    EXPORT_CONSTANT(DB_ABORT);
    EXPORT_CONSTANT(DB_SUSPEND);
    EXPORT_CONSTANT(DB_RESUME);

    /* Errors */
    EXPORT_CONSTANT(E_NOERROR);
    EXPORT_CONSTANT(E_BADFTYPE);
    EXPORT_CONSTANT(E_NOTIMP);
    EXPORT_CONSTANT(E_NOFILE);
    EXPORT_CONSTANT(E_INTERNAL);
    EXPORT_CONSTANT(E_NOMEM);
    EXPORT_CONSTANT(E_BADARGS);
    EXPORT_CONSTANT(E_CALLFAIL);
    EXPORT_CONSTANT(E_NOTFOUND);
    EXPORT_CONSTANT(E_TAURSTATE);
    EXPORT_CONSTANT(E_MSERVER);
    EXPORT_CONSTANT(E_PROTO     );
    EXPORT_CONSTANT(E_NOTDIR);
    EXPORT_CONSTANT(E_MAXOPEN);
    EXPORT_CONSTANT(E_NOTFILTER);
    EXPORT_CONSTANT(E_MAXFILTERS);
    EXPORT_CONSTANT(E_FEXIST);
    EXPORT_CONSTANT(E_FILEISDIR);
    EXPORT_CONSTANT(E_FILENOREAD);
    EXPORT_CONSTANT(E_SYSTEMERR);
    EXPORT_CONSTANT(E_FILENOWRITE);
    EXPORT_CONSTANT(E_INVALIDNAME);
    EXPORT_CONSTANT(E_NOOVERWRITE);
    EXPORT_CONSTANT(E_CHECKSUM);
    EXPORT_CONSTANT(E_NERRORS);

    /* Definitions for MAJOR_ORDER */
    EXPORT_CONSTANT(DB_ROWMAJOR);
    EXPORT_CONSTANT(DB_COLMAJOR);

    /* Definitions for COORD_TYPE */
    EXPORT_CONSTANT(DB_COLLINEAR);
    EXPORT_CONSTANT(DB_NONCOLLINEAR);
    EXPORT_CONSTANT(DB_QUAD_RECT);
    EXPORT_CONSTANT(DB_QUAD_CURV);

    /* Definitions for CENTERING */
    EXPORT_CONSTANT(DB_NOTCENT);
    EXPORT_CONSTANT(DB_NODECENT);
    EXPORT_CONSTANT(DB_ZONECENT);
    EXPORT_CONSTANT(DB_FACECENT);
    EXPORT_CONSTANT(DB_BNDCENT);

    /* Definitions for COORD_SYSTEM */
    EXPORT_CONSTANT(DB_CARTESIAN);
    EXPORT_CONSTANT(DB_CYLINDRICAL);
    EXPORT_CONSTANT(DB_SPHERICAL);
    EXPORT_CONSTANT(DB_NUMERICAL);
    EXPORT_CONSTANT(DB_OTHER);

    /* Definitions for ZONE FACE_TYPE */
    EXPORT_CONSTANT(DB_RECTILINEAR);
    EXPORT_CONSTANT(DB_CURVILINEAR);

    /* Definitions for PLANAR */
    EXPORT_CONSTANT(DB_AREA);
    EXPORT_CONSTANT(DB_VOLUME);
    /* Definitions for flag values */
    EXPORT_CONSTANT(DB_ON);
    EXPORT_CONSTANT(DB_OFF);

    /* Definitions for derived variable types */
    EXPORT_CONSTANT(DB_VARTYPE_SCALAR);
    EXPORT_CONSTANT(DB_VARTYPE_VECTOR);
    EXPORT_CONSTANT(DB_VARTYPE_TENSOR);
    EXPORT_CONSTANT(DB_VARTYPE_SYMTENSOR);
    EXPORT_CONSTANT(DB_VARTYPE_ARRAY);
    EXPORT_CONSTANT(DB_VARTYPE_MATERIAL);
    EXPORT_CONSTANT(DB_VARTYPE_SPECIES);
    EXPORT_CONSTANT(DB_VARTYPE_LABEL);

    /* Definitions for CSG boundary types */
    EXPORT_CONSTANT(DBCSG_QUADRIC_G);
    EXPORT_CONSTANT(DBCSG_SPHERE_PR);
    EXPORT_CONSTANT(DBCSG_ELLIPSOID_PRRR);
    EXPORT_CONSTANT(DBCSG_PLANE_G);
    EXPORT_CONSTANT(DBCSG_PLANE_X);
    EXPORT_CONSTANT(DBCSG_PLANE_Y);
    EXPORT_CONSTANT(DBCSG_PLANE_Z);
    EXPORT_CONSTANT(DBCSG_PLANE_PN);
    EXPORT_CONSTANT(DBCSG_PLANE_PPP);
    EXPORT_CONSTANT(DBCSG_CYLINDER_PNLR);
    EXPORT_CONSTANT(DBCSG_CYLINDER_PPR);
    EXPORT_CONSTANT(DBCSG_BOX_XYZXYZ);
    EXPORT_CONSTANT(DBCSG_CONE_PNLA);
    EXPORT_CONSTANT(DBCSG_CONE_PPA);
    EXPORT_CONSTANT(DBCSG_POLYHEDRON_KF);
    EXPORT_CONSTANT(DBCSG_HEX_6F);
    EXPORT_CONSTANT(DBCSG_TET_4F);
    EXPORT_CONSTANT(DBCSG_PYRAMID_5F);
    EXPORT_CONSTANT(DBCSG_PRISM_5F);

    /* Definitions for 2D CSG boundary types */
    EXPORT_CONSTANT(DBCSG_QUADRATIC_G);
    EXPORT_CONSTANT(DBCSG_CIRCLE_PR);
    EXPORT_CONSTANT(DBCSG_ELLIPSE_PRR);
    EXPORT_CONSTANT(DBCSG_LINE_G);
    EXPORT_CONSTANT(DBCSG_LINE_X);
    EXPORT_CONSTANT(DBCSG_LINE_Y);
    EXPORT_CONSTANT(DBCSG_LINE_PN);
    EXPORT_CONSTANT(DBCSG_LINE_PP);
    EXPORT_CONSTANT(DBCSG_BOX_XYXY);
    EXPORT_CONSTANT(DBCSG_ANGLE_PNLA);
    EXPORT_CONSTANT(DBCSG_ANGLE_PPA);
    EXPORT_CONSTANT(DBCSG_POLYGON_KP);
    EXPORT_CONSTANT(DBCSG_TRI_3P);
    EXPORT_CONSTANT(DBCSG_QUAD_4P);

    /* Definitions for CSG Region operators */
    EXPORT_CONSTANT(DBCSG_INNER);
    EXPORT_CONSTANT(DBCSG_OUTER);
    EXPORT_CONSTANT(DBCSG_ON);
    EXPORT_CONSTANT(DBCSG_UNION);
    EXPORT_CONSTANT(DBCSG_INTERSECT);
    EXPORT_CONSTANT(DBCSG_DIFF);
    EXPORT_CONSTANT(DBCSG_COMPLIMENT);
    EXPORT_CONSTANT(DBCSG_XFORM);
    EXPORT_CONSTANT(DBCSG_SWEEP);
#undef EXPORT_CONSTANT
    return result;
  }

#define CALL_GUARDED(NAME, ARGLIST) \
  if (NAME ARGLIST) \
    throw std::runtime_error(#NAME " failed");

#define CONVERT_INT_LIST(NAME) \
  std::vector<int> NAME; \
  for (unsigned i = 0; i < len(NAME##_py); i++) \
      NAME.push_back(extract<int>(NAME##_py[i]));

  class DBoptlistWrapper : boost::noncopyable
  {
    private:
      DBoptlist *m_optlist;
      boost::scoped_array<char> m_option_storage;
      unsigned m_option_storage_size;
      unsigned m_option_storage_occupied;

    public:
      DBoptlistWrapper(unsigned maxsize, unsigned storage_size)
        : m_optlist(DBMakeOptlist(maxsize)),
        m_option_storage(new char[storage_size]),
        m_option_storage_size(storage_size),
        m_option_storage_occupied(0)
      { 
        if (m_optlist == NULL)
          throw std::runtime_error("DBMakeOptlist failed");
      }
      ~DBoptlistWrapper()
      {
        CALL_GUARDED(DBFreeOptlist, (m_optlist));
      }

      void add_option(int option, int value)
      {
        CALL_GUARDED(DBAddOption,(m_optlist, option, 
              add_storage_data((void *) &value, sizeof(value))
              ));
      }

      void add_option(int option, double value)
      {
        switch (option)
        {
          case DBOPT_DTIME:
            {
              CALL_GUARDED(DBAddOption,(m_optlist, option, 
                    add_storage_data((void *) &value, sizeof(value))
                    ));
              break;
            }
          default:
            {
              float cast_val = value;
              CALL_GUARDED(DBAddOption,(m_optlist, option, 
                    add_storage_data((void *) &cast_val, sizeof(cast_val))
                    ));
              break;
            }
        }
      }

      void add_option(int option, const std::string &value)
      {
        CALL_GUARDED(DBAddOption,(m_optlist, option, 
              add_storage_data((void *) value.data(), value.size()+1)
              ));
      }

      DBoptlist *get_optlist()
      {
        return m_optlist;
      }

    protected:
      void *add_storage_data(void *data, unsigned size)
      {
        if (m_option_storage_occupied + size > m_option_storage_size)
          throw std::runtime_error("silo option list storage exhausted"
              "--specify bigger storage size");

        void *dest = m_option_storage.get() + m_option_storage_occupied;
        memcpy(dest, data, size);
        m_option_storage_occupied += size;
        return dest;
      }

  };

  class DBfileWrapper : boost::noncopyable
  {
    public:
      DBfileWrapper(const char *name, int target, int mode)
        : m_dbfile(DBOpen(name, target, mode))
      { 
        if (m_dbfile == NULL)
          throw std::runtime_error("DBOpen failed");
      }
      DBfileWrapper(const char *name, int mode, int target, const char *info, int type)
        : m_dbfile(DBCreate(name, mode, target, info, type))
      { 
        if (m_dbfile == NULL)
          throw std::runtime_error("DBCreate failed");
      }

      ~DBfileWrapper()
      {
        CALL_GUARDED(DBClose, (m_dbfile));
      }

      operator DBfile *()
      {
        return m_dbfile;
      }




      void put_zonelist(const char *name, int nzones, int ndims,
          object nodelist_py, object shapesize_py,
          object shapecounts_py)
      {
        CONVERT_INT_LIST(nodelist);
        CONVERT_INT_LIST(shapesize);
        CONVERT_INT_LIST(shapecounts);
        CALL_GUARDED(DBPutZonelist, (m_dbfile, name, nzones, ndims, nodelist.data(),
            len(nodelist_py), 0, shapesize.data(), shapecounts.data(),
            len(shapesize_py)));
      }




      void put_ucdmesh(const char *name, int ndims,
             object coordnames_py, object coords_py, 
             int nzones, const char *zonel_name, const char *facel_name,
             DBoptlistWrapper &optlist)
      {
        typedef double value_type;
        int datatype = DB_DOUBLE;

        int nnodes = len(coords_py)/ndims;
        std::vector<value_type> coords;
        for (unsigned d = 0; d < ndims; d++)
          for (unsigned i = 0; i < nnodes; i++)
            coords.push_back(extract<value_type>(coords_py[d*nnodes+i]));

        std::vector<value_type *> coord_starts;
        for (unsigned d = 0; d < ndims; d++)
          coord_starts.push_back(coords.data()+d*nnodes);

        CALL_GUARDED(DBPutUcdmesh, (m_dbfile, name, ndims, 
            /* coordnames*/ NULL,
            (float **) coord_starts.data(), nnodes,
            nzones, zonel_name, facel_name,
            datatype, optlist.get_optlist()));
      }




      void put_ucdvar1(const char *vname, const char *mname, hedge::vector &v,
             /*float *mixvar, int mixlen, */int centering,
             DBoptlistWrapper &optlist)
      {
        typedef hedge::vector::value_type value_type;
        int datatype = DB_DOUBLE; // FIXME: should depend on real data type

        CALL_GUARDED(DBPutUcdvar1, (m_dbfile, vname, mname, 
            (float *) traits::vector_storage(v),
            v.size(), 
            /* mixvar */ NULL, /* mixlen */ 0, 
            datatype, centering,
            optlist.get_optlist()));
      }




      void put_ucdvar(const char *vname, const char *mname, 
          object varnames_py, object vars_py, 
          /*float *mixvars[], int mixlen,*/ 
          int centering, 
          DBoptlistWrapper &optlist)
      {
        typedef hedge::vector::value_type value_type;
        int datatype = DB_DOUBLE; // FIXME: should depend on real data type

        if (len(varnames_py) != len(vars_py))
          PYTHON_ERROR(ValueError, "varnames and vars must have the same length");

        std::vector<std::string> varnames_container;
        std::vector<const char *> varnames;
        for (unsigned i = 0; i < len(varnames_py); i++)
          varnames_container.push_back(
              extract<std::string>(varnames_py[i]));
        for (unsigned i = 0; i < len(varnames_py); i++)
          varnames.push_back(varnames_container[i].data());

        std::vector<float *> vars;
        bool first = true;
        unsigned vlength = 0;
        for (unsigned i = 0; i < len(vars_py); i++)
        {
          hedge::vector &v = extract<hedge::vector &>(vars_py[i]);
          if (first)
          {
            vlength = v.size();
            first = false;
          }
          else if (vlength != v.size())
            PYTHON_ERROR(ValueError, "field components need to have matching lengths");
          vars.push_back((float *) traits::vector_storage(v));
        }

        CALL_GUARDED(DBPutUcdvar, (m_dbfile, vname, mname, 
            len(vars_py), (char **) varnames.data(), vars.data(), 
            vlength, 
            /* mixvar */ NULL, /* mixlen */ 0, 
            datatype, centering, optlist.get_optlist()));
      }




      void put_defvars(std::string id, object vars_py)
      {
        std::vector<std::string> varnames_container;
        std::vector<const char *> varnames;
        std::vector<std::string> vardefs_container;
        std::vector<const char *> vardefs;
        std::vector<int> vartypes;
        std::vector<DBoptlist *> varopts;

        for (unsigned i = 0; i < len(vars_py); i++)
        {
          object entry = vars_py[i];
          varnames_container.push_back(extract<std::string>(entry[0]));
          vardefs_container.push_back(extract<std::string>(entry[1]));
          if (len(entry) == 2)
            vartypes.push_back(DB_VARTYPE_SCALAR);
          else 
          {
            vartypes.push_back(extract<int>(entry[2]));
            if (len(entry) == 4)
              varopts.push_back(extract<DBoptlistWrapper *>(entry[3])()->get_optlist());
            else
              varopts.push_back(NULL);
          }
        }

        for (unsigned i = 0; i < len(vars_py); i++)
        {
          varnames.push_back(varnames_container[i].data());
          vardefs.push_back(vardefs_container[i].data());
        }

        CALL_GUARDED(DBPutDefvars, (m_dbfile, id.data(), len(vars_py), 
            varnames.data(), vartypes.data(), vardefs.data(), varopts.data()));
      }




      void put_pointmesh(const char *id, int ndims, object coords_py,
          DBoptlistWrapper &optlist)
      {
        typedef double value_type;
        std::vector<value_type> coords;
        int datatype = DB_DOUBLE; // FIXME: should depend on real data type

        int npoints = len(coords_py)/ndims;

        for (unsigned d = 0; d < ndims; d++)
          for (unsigned i = 0; i < npoints; i++)
            coords.push_back(extract<value_type>(coords_py[d*npoints+i]));

        std::vector<float *> coord_starts;
        for (unsigned d = 0; d < ndims; d++)
          coord_starts.push_back((float *) (coords.data()+d*npoints));

        CALL_GUARDED(DBPutPointmesh, (m_dbfile, id, 
              ndims, coord_starts.data(), npoints, datatype, 
              optlist.get_optlist()));
      }




      void put_pointvar1(const char *vname, const char *mname, 
          hedge::vector &v,
          DBoptlistWrapper &optlist)
      {
        int datatype = DB_DOUBLE; // FIXME: should depend on real data type

        CALL_GUARDED(DBPutPointvar1, (m_dbfile, vname, mname,
              (float *) traits::vector_storage(v), v.size(), datatype,
              optlist.get_optlist()));
      }




      void put_pointvar(const char *vname, const char *mname, 
          object vars_py,
          DBoptlistWrapper &optlist)
      {
        int datatype = DB_DOUBLE; // FIXME: should depend on real data type

        std::vector<float *> vars;
        bool first = true;
        unsigned vlength = 0;
        for (unsigned i = 0; i < len(vars_py); i++)
        {
          hedge::vector &v = extract<hedge::vector &>(vars_py[i]);
          if (first)
          {
            vlength = v.size();
            first = false;
          }
          else if (vlength != v.size())
            PYTHON_ERROR(ValueError, "field components need to have matching lengths");

          vars.push_back((float *) traits::vector_storage(v));
        }

        CALL_GUARDED(DBPutPointvar, (m_dbfile, vname, mname,
              len(vars_py), vars.data(), vlength, datatype,
              optlist.get_optlist()));
      }
    private:
      DBfile *m_dbfile;
  };

#endif
}




BOOST_PYTHON_MODULE(_silo)
{
  DEF_SIMPLE_FUNCTION(have_silo);

#ifdef USE_SILO
  DEF_SIMPLE_FUNCTION(symbols);

  enum_<DBdatatype>("DBdatatype")
    .ENUM_VALUE(DB_INT)
    .ENUM_VALUE(DB_SHORT)
    .ENUM_VALUE(DB_LONG)
    .ENUM_VALUE(DB_FLOAT)
    .ENUM_VALUE(DB_DOUBLE)
    .ENUM_VALUE(DB_CHAR)
    .ENUM_VALUE(DB_NOTYPE)
    ;

  {
    typedef DBfileWrapper cl;
    class_<cl, boost::noncopyable>("DBFile", init<const char *, int, int>())
      .def(init<const char *, int, int, const char *, int>())
      .DEF_SIMPLE_METHOD(put_zonelist)
      .DEF_SIMPLE_METHOD(put_ucdmesh)
      .DEF_SIMPLE_METHOD(put_ucdvar1)
      .DEF_SIMPLE_METHOD(put_ucdvar)
      .DEF_SIMPLE_METHOD(put_defvars)
      .DEF_SIMPLE_METHOD(put_pointmesh)
      .DEF_SIMPLE_METHOD(put_pointvar1)
      .DEF_SIMPLE_METHOD(put_pointvar)
      ;
  }

  {
    typedef DBoptlistWrapper cl;
    class_<cl, boost::noncopyable>("DBOptlist", init<unsigned, unsigned>())
      .def("add_int_option", (void (cl::*)(int, int)) &cl::add_option)
      .def("add_option", (void (cl::*)(int, double)) &cl::add_option)
      .def("add_option", (void (cl::*)(int, const std::string &)) &cl::add_option)
      ;
  }
#endif
}
