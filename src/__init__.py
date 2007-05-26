# Blank.



if __name__ == "__main__":
    from hedge.element import Triangle
    outf = file("nodes.dat", "w")
    tri = Triangle(33)
    for x in tri.nodes():
        outf.write("%f\t%f\n" % tuple(tri.equilateral_to_unit(x)))
