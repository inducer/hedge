<TeXmacs|1.0.7.2>

<style|generic>

<\body>
  Variational formulation of Poisson with fluxes <math|<wide|u|^><rsub|h>>
  and <math|<wide|\<b-sigma\>|^><rsub|h>>, solving for
  <math|u<rsub|h>,\<b-sigma\><rsub|h>>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<big|int><rsub|\<Omega\>>\<b-sigma\><rsub|h>\<cdot\>\<b-tau\><rsub|h>>|<cell|=>|<cell|-<big|int><rsub|\<Omega\>>u<rsub|h>\<nabla\>\<cdot\>\<b-tau\><rsub|h>+<big|sum><rsub|K><big|int><rsub|\<partial\>K><wide|u|^><rsub|h>\<b-tau\><rsub|h>\<cdot\>\<b-n\>,>>|<row|<cell|<big|int><rsub|\<Omega\>>\<b-sigma\><rsub|h>\<cdot\>\<nabla\>v<rsub|h>>|<cell|=>|<cell|<big|int><rsub|\<Omega\>>v<rsub|h>f+<big|int><rsub|\<partial\>K>v<rsub|h><wide|\<b-sigma\>|^><rsub|h>\<cdot\>\<b-n\>.>>>>
  </eqnarray*>

  Use

  <\equation*>
    <big|sum><rsub|K><big|int><rsub|\<partial\>K>v<rsub|k>\<b-tau\><rsub|K>\<cdot\>\<b-n\><rsub|K>=<big|int><rsub|\<Gamma\>>[v]\<cdot\>{\<b-tau\>}+<big|int><rsub|\<Gamma\><rsub|0>>{v}[\<b-tau\>],
  </equation*>

  where <math|\<Gamma\><rsub|0>> includes the interior faces, to convert to:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<big|int><rsub|\<Omega\>>\<b-sigma\><rsub|h>\<cdot\>\<b-tau\><rsub|h>>|<cell|=>|<cell|-<big|int><rsub|\<Omega\>>u<rsub|h>\<nabla\>\<cdot\>\<b-tau\><rsub|h>+<big|int><rsub|\<Gamma\>>[<wide|u|^><rsub|h>]\<cdot\>{\<b-tau\><rsub|h>}+<big|int><rsub|\<Gamma\><rsub|0>>{<wide|u|^><rsub|h>}[\<b-tau\><rsub|h>],>>|<row|<cell|<big|int><rsub|\<Omega\>>\<b-sigma\><rsub|h>\<cdot\>\<nabla\>v<rsub|h>>|<cell|=>|<cell|<big|int><rsub|\<Omega\>>v<rsub|h>f+<big|int><rsub|\<Gamma\>>[v<rsub|h>]\<cdot\>{<wide|\<b-sigma\>|^><rsub|h>}+<big|int><rsub|\<Gamma\><rsub|0>>{v<rsub|h>}[<wide|\<b-sigma\>|^><rsub|h>].>>>>
  </eqnarray*>

  Now pick <math|\<b-tau\><rsub|h>\<assign\>\<nabla\>v<rsub|h>> to facilitate
  conversion to a bilinear form. This turns the first equation into.

  <\eqnarray*>
    <tformat|<table|<row|<cell|<big|int><rsub|\<Omega\>>\<b-sigma\><rsub|h>\<cdot\>\<nabla\>v<rsub|h>>|<cell|=>|<cell|-<big|int><rsub|\<Omega\>>u<rsub|h>\<nabla\>\<cdot\>(\<nabla\>v<rsub|h>)+<big|int><rsub|\<Gamma\>>[<wide|u|^><rsub|h>]\<cdot\>{\<nabla\>v<rsub|h>}+<big|int><rsub|\<Gamma\><rsub|0>>{<wide|u|^><rsub|h>}[\<nabla\>v<rsub|h>].>>>>
  </eqnarray*>

  We can now equate the RHSs of this latter equation and the second equation
  above:

  <\equation*>
    <big|int><rsub|\<Omega\>>v<rsub|h>f+<big|int><rsub|\<Gamma\>>[v<rsub|h>]\<cdot\>{<wide|\<b-sigma\>|^><rsub|h>}+<big|int><rsub|\<Gamma\><rsub|0>>{v<rsub|h>}[<wide|\<b-sigma\>|^><rsub|h>]=-<wide*|<big|int><rsub|\<Omega\>>u<rsub|h>\<nabla\>\<cdot\>(\<nabla\>v<rsub|h>)|\<wide-underbrace\>><rsub|A>+<big|int><rsub|\<Gamma\>>[<wide|u|^><rsub|h>]\<cdot\>{\<nabla\>v<rsub|h>}+<big|int><rsub|\<Gamma\><rsub|0>>{<wide|u|^><rsub|h>}[\<nabla\>v<rsub|h>].
  </equation*>

  Now use the integration-by-parts formula

  <\equation*>
    <big|int><rsub|\<Omega\>>\<nabla\><rsub|h>v\<cdot\>\<b-tau\>+<big|int><rsub|\<Omega\>>v\<nabla\>\<cdot\>\<b-tau\>=<big|int><rsub|\<Gamma\>>[v]\<cdot\>{\<b-tau\>}+<big|int><rsub|\<Gamma\><rsub|0>>{v}[\<b-tau\>]
  </equation*>

  to eliminate the double derivative:

  <\equation*>
    A=<big|int><rsub|\<Omega\>>u<rsub|h>\<nabla\>\<cdot\>(\<nabla\>v<rsub|h>)=<with|color|blue|-<big|int><rsub|\<Omega\>>\<nabla\>u<rsub|h>\<cdot\>\<nabla\>v<rsub|h>+<big|int><rsub|\<Gamma\>>[u<rsub|h>]\<cdot\>{\<nabla\>v<rsub|h>}+<big|int><rsub|\<Gamma\><rsub|0>>{u<rsub|h>}[\<nabla\>v<rsub|h>]>
  </equation*>

  and obtain

  <\equation*>
    <big|int><rsub|\<Omega\>>v<rsub|h>f+<big|int><rsub|\<Gamma\>>[v<rsub|h>]\<cdot\>{<wide|\<b-sigma\>|^><rsub|h>}+<big|int><rsub|\<Gamma\><rsub|0>>{v<rsub|h>}[<wide|\<b-sigma\>|^><rsub|h>]=<with|color|blue|<big|int><rsub|\<Omega\>>\<nabla\>u<rsub|h>\<cdot\>\<nabla\>v<rsub|h>-<big|int><rsub|\<Gamma\>>[u<rsub|h>]\<cdot\>{\<nabla\>v<rsub|h>}-<big|int><rsub|\<Gamma\><rsub|0>>{u<rsub|h>}[\<nabla\>v<rsub|h>]>+<big|int><rsub|\<Gamma\>>[<wide|u|^><rsub|h>]\<cdot\>{\<nabla\>v<rsub|h>}+<big|int><rsub|\<Gamma\><rsub|0>>{<wide|u|^><rsub|h>}[\<nabla\>v<rsub|h>]
  </equation*>

  Regroup and reorder:

  <\equation*>
    <big|int><rsub|\<Omega\>>\<nabla\>u<rsub|h>\<cdot\>\<nabla\>v<rsub|h>+<big|int><rsub|\<Gamma\><rsub|0>>{<wide|u|^><rsub|h>-u<rsub|h>}[\<nabla\>v<rsub|h>]+<big|int><rsub|\<Gamma\>>[<wide|u|^><rsub|h>-u<rsub|h>]\<cdot\>{\<nabla\>v<rsub|h>}-<big|int><rsub|\<Gamma\>>[v<rsub|h>]\<cdot\>{<wide|\<b-sigma\>|^><rsub|h>}-<big|int><rsub|\<Gamma\><rsub|0>>{v<rsub|h>}[<wide|\<b-sigma\>|^><rsub|h>]=<big|int><rsub|\<Omega\>>v<rsub|h>f.
  </equation*>

  For IP, e.g.

  <\equation*>
    <wide|u|^><rsub|h>\<assign\>,<space|2em><wide|\<b-sigma\>|^><rsub|h>\<assign\>{\<nabla\><rsub|h>u<rsub|h>}-<frac|\<eta\>|h<rsub|e>>[u<rsub|h>].
  </equation*>

  Substitute in:

  <\equation*>
    <big|int><rsub|\<Omega\>>\<nabla\>u<rsub|h>\<cdot\>\<nabla\>v<rsub|h>+<wide*|<big|int><rsub|\<Gamma\><rsub|0>>{<with|color|blue|{u<rsub|h>}>-u<rsub|h>}[\<nabla\>v<rsub|h>]|\<wide-underbrace\>><rsub|0>+<big|int><rsub|\<Gamma\>>[<with|color|blue|{u<rsub|h>}>-u<rsub|h>]\<cdot\>{\<nabla\>v<rsub|h>}-<big|int><rsub|\<Gamma\>>[v<rsub|h>]\<cdot\>{<with|color|blue|{\<nabla\><rsub|h>u<rsub|h>}-<frac|\<eta\>|h<rsub|e>>[u<rsub|h>]}>-<big|int><rsub|\<Gamma\><rsub|0>>{v<rsub|h>}[<with|color|blue|{\<nabla\><rsub|h>u<rsub|h>}-<frac|\<eta\>|h<rsub|e>>[u<rsub|h>]>]=<big|int><rsub|\<Omega\>>v<rsub|h>f
  </equation*>

  and obtain:

  <\equation*>
    <big|int><rsub|\<Omega\>>\<nabla\>u<rsub|h>\<cdot\>\<nabla\>v<rsub|h>-<big|int><rsub|\<Gamma\>>[u<rsub|h>]\<cdot\>{\<nabla\>v<rsub|h>}-<big|int><rsub|\<Gamma\>>[v<rsub|h>]\<cdot\>{\<nabla\><rsub|h>u<rsub|h>}-<big|int><rsub|\<Gamma\>>[v<rsub|h>]<frac|\<eta\>|h<rsub|e>>[u<rsub|h>]=<big|int><rsub|\<Omega\>>v<rsub|h>f.
  </equation*>
</body>

<\initial>
  <\collection>
    <associate|page-type|letter>
  </collection>
</initial>