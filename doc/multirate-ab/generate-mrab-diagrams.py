from hedge.timestep.multirate_ab.methods import methods
from hedge.timestep.multirate_ab.processors import MRABToTeXProcessor

for name, method in methods.iteritems():
    mrab2tex = MRABToTeXProcessor(method, 3)
    mrab2tex.run()
    open("out/%s.tex" % name, "w").write(
            "Scheme name: \\verb|%s|\n\n" % name+
            mrab2tex.get_result())

