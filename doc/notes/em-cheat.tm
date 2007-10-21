<TeXmacs|1.0.6>

<style|generic>

<\body>
  <strong|<center|Electromagnetics Cheat Sheet>>

  1. Units:

  <\eqnarray*>
    <tformat|<table|<row|<cell|[E]>|<cell|=>|<cell|<frac|N|C>>>|<row|<cell|[H]>|<cell|=>|<cell|<frac|A|m>=<frac|C|m\<cdot\>s>>>|<row|<cell|[\<mu\><rsub|0>]>|<cell|=>|<cell|<frac|N|A<rsup|2>>>>|<row|<cell|[\<varepsilon\><rsub|0>]>|<cell|=>|<cell|<frac|s<rsup|2>A<rsup|2>|m<rsup|2>N<rsup|2>>>>|<row|<cell|[Z]>|<cell|=>|<cell|<frac|N*m<rsup|2>|A<rsup|2>s<rsup|2>>>>|<row|<cell|[Y]>|<cell|=>|<cell|<frac|A<rsup|2>s<rsup|2>|N*m<rsup|2>>>>>>
  </eqnarray*>

  2. Equations

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|D|\<dot\>>>|<cell|=>|<cell|\<nabla\>\<times\>H+J>>|<row|<cell|<wide|B|\<dot\>>>|<cell|=>|<cell|-\<nabla\>\<times\>E>>|<row|<cell|\<nabla\>\<cdot\>D>|<cell|=>|<cell|\<rho\>>>|<row|<cell|\<nabla\>\<cdot\>B>|<cell|=>|<cell|0>>>>
  </eqnarray*>

  <\equation*>
    B=\<mu\>H,<space|1em>D=\<varepsilon\>E
  </equation*>

  <\equation*>
    Z=<sqrt|<frac|\<mu\>|\<varepsilon\>>>,<space|1em>Y=<sqrt|<frac|\<varepsilon\>|\<mu\>>>.
  </equation*>

  3. Boundary conditions

  <\equation*>
    <with|mode|text|PEC:><space|1em>n\<times\>E=0,<space|1em>n\<cdot\>H=0.
  </equation*>

  4. Cross product, curl

  <\equation*>
    <matrix|<tformat|<table|<row|<cell|a<rsub|1>>>|<row|<cell|a<rsub|2>>>|<row|<cell|a<rsub|3>>>>>>\<times\><matrix|<tformat|<table|<row|<cell|b<rsub|1>>>|<row|<cell|b<rsub|2>>>|<row|<cell|b<rsub|3>>>>>>=<matrix|<tformat|<table|<row|<cell|a<rsub|2>b<rsub|3>-a<rsub|3>b<rsub|2>>>|<row|<cell|a<rsub|3>b<rsub|1>-a<rsub|1>b<rsub|3>>>|<row|<cell|a<rsub|1>b<rsub|2>-a<rsub|2>b<rsub|1>>>>>>,<space|1em>curl
    F=<matrix|<tformat|<table|<row|<cell|\<partial\><rsub|2>F<rsub|3>-\<partial\><rsub|3>F<rsub|2>>>|<row|<cell|\<partial\><rsub|3>F<rsub|1>-\<partial\><rsub|1>F<rsub|3>>>|<row|<cell|\<partial\><rsub|1>F<rsub|2>-\<partial\><rsub|2>F<rsub|1>>>>>>
  </equation*>

  <\equation*>
    a\<times\>(b\<times\>c)=b(a\<cdot\>c)-c(a\<cdot\>b)
  </equation*>
</body>

<\initial>
  <\collection>
    <associate|page-type|letter>
  </collection>
</initial>