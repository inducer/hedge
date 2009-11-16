from hedge.discretization.local import TetrahedronDiscretization as Tet
from pytools import Table

tbl = Table()
tbl.add_row(("N", "DOFs", "Face DOFs", "Total Face DOFs"))
for order in range(1, 9):
    t = Tet(order)
    row = (order, t.node_count(), t.face_node_count(), 
        t.face_node_count() * t.face_count())
    tbl.add_row(row)
    print " & ".join(str(x) for x in row), r"\\"


print tbl
