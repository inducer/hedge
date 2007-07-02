#include <boost/numeric/bindings/traits/traits.hpp>
#include <vector>
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

#define CONVERT_INT_LIST(NAME) \
  std::vector<int> NAME; \
  for (unsigned i = 0; i < len(NAME##_py); i++) \
      NAME.push_back(extract<int>(NAME##_py[i]));

  class DBfileWrapper
  {
    public:
      DBfileWrapper(const char *name, int target, int mode)
        : m_dbfile(DBOpen(name, target, mode))
      { }
      DBfileWrapper(const char *name, int mode, int target, const char *info, int type)
        : m_dbfile(DBCreate(name, mode, target, info, type))
      { }

      ~DBfileWrapper()
      {
        DBClose(m_dbfile);
      }

      operator DBfile *()
      {
        return m_dbfile;
      }

      int PutZonelist(const char *name, int nzones, int ndims,
          object nodelist_py, object shapesize_py,
          object shapecounts_py)
      {
        CONVERT_INT_LIST(nodelist);
        CONVERT_INT_LIST(shapesize);
        CONVERT_INT_LIST(shapecounts);
        return DBPutZonelist(m_dbfile, name, nzones, ndims, nodelist.data(),
            len(nodelist_py), 0, shapesize.data(), shapecounts.data(),
            len(shapesize_py));
      }

      int PutUcdmesh(const char *name, int ndims,
             object coordnames_py, object coords_py, 
             int nzones, const char *zonel_name, const char *facel_name
             /*, DBoptlist *optlist*/)
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

        return DBPutUcdmesh(m_dbfile, name, ndims, 
            /* coordnames*/ NULL,
            (float **) coord_starts.data(), nnodes,
            nzones, zonel_name, facel_name,
            datatype, NULL);
      }

      int PutUcdvar1(const char *vname, const char *mname, hedge::vector &v,
             int nels, /*float *mixvar, int mixlen, */int centering
             /*, DBoptlist *optlist*/)
      {
        typedef hedge::vector::value_type value_type;
        int datatype = DB_DOUBLE; // FIXME: sketchy

        return DBPutUcdvar1(m_dbfile, vname, mname, 
            (float *) traits::vector_storage(v),
            nels, 
            /* mixvar */ NULL, /* mixlen */ 0, 
            datatype, centering,
            /* optlist */ NULL);
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
    class_<cl, boost::noncopyable>("DBfile", init<const char *, int, int>())
      .def(init<const char *, int, int, const char *, int>())
      .DEF_SIMPLE_METHOD(PutZonelist)
      .DEF_SIMPLE_METHOD(PutUcdmesh)
      .DEF_SIMPLE_METHOD(PutUcdvar1)
      ;
  }
#endif
}
