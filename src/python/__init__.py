"""
Hedge
-----

Hybrid'n'Easy Discontinuous Galerkin Environment
"""



if __name__ == "__main__":
    from hedge.element import Triangle
    tri = Triangle(17)
    #print tri.vandermonde()
    outf = file("nodes.dat", "w")
    nodes = list(tri.unit_nodes())
    face_idx = list(tri.face_indices())
    for fi, i in enumerate(face_idx[0]+face_idx[1]+face_idx[2]):
        outf.write("%f\t%f\t%f\n" % (nodes[i][0], nodes[i][1], fi))
    from hedge.polynomial import legendre_polynomial
    from pymbolic import differentiate, var

    #lp = legendre_polynomial(17)
    #print lp
    #print repr(lp.data[0][1])
    #print differentiate(lp, var("x"))
