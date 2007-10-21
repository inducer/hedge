<TeXmacs|1.0.6>

<style|generic>

<\body>
  <center|<strong|The 3D PML in all its gory detail>>

  <center|Andreas Klöckner <with|font-family|tt|\<less\>andreas@mcs.anl.gov\<gtr\>>>

  Define the PML Material tensor:

  <\equation*>
    <wide|S|~>\<assign\>S<rsup|-1><matrix|<tformat|<table|<row|<cell|s<rsub|y>s<rsub|z>>|<cell|>|<cell|>>|<row|<cell|>|<cell|s<rsub|x>s<rsub|z>>|<cell|>>|<row|<cell|>|<cell|>|<cell|s<rsub|x>s<rsub|y>>>>>>,<space|1em>S\<assign\><matrix|<tformat|<table|<row|<cell|s<rsub|x>>|<cell|>|<cell|>>|<row|<cell|>|<cell|s<rsub|y>>|<cell|>>|<row|<cell|>|<cell|>|<cell|s<rsub|z>>>>>>.
  </equation*>

  with

  <\equation*>
    s<rsub|i>=1+<frac|\<sigma\><rsub|i>|i\<omega\>\<varepsilon\>><space|1em>(i=x,y,z).
  </equation*>

  In a simple case, say, where only <with|mode|math|s<rsub|x>\<neq\>1>, we
  obtain

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|S|~><rsub|x,x>=<frac|1|1+<frac|\<sigma\><rsub|x>|i\<omega\>\<varepsilon\>>>=<frac|1|<frac|i\<omega\>\<varepsilon\>+\<sigma\><rsub|x>|i\<omega\>\<varepsilon\>>>=<frac|i\<omega\>\<varepsilon\>|i\<omega\>\<varepsilon\>+\<sigma\><rsub|x>>>|<cell|>|<cell|>>>>
  </eqnarray*>

  In a more complicated case, with, say, only <with|mode|math|s<rsub|z>=1>,
  we have

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|S|~><rsub|x,x>=<frac|s<rsub|y>|s<rsub|x>>>|<cell|=>|<cell|<frac|1+<frac|\<sigma\><rsub|y>|i\<omega\>\<varepsilon\>>|1+<frac|\<sigma\><rsub|x>|i\<omega\>\<varepsilon\>>>=<frac|i\<omega\>\<varepsilon\>+\<sigma\><rsub|y>|i\<omega\>\<varepsilon\>+\<sigma\><rsub|x>>.>>>>
  </eqnarray*>

  In the most complicated example with all
  <with|mode|math|s<rsub|i>\<neq\>1>, we get

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|S|~><rsub|x,x>=<frac|s<rsub|y>s<rsub|z>|s<rsub|x>>>|<cell|=>|<cell|<frac|<left|(>1+<frac|\<sigma\><rsub|y>|i\<omega\>\<varepsilon\>><right|)><left|(>1+<frac|\<sigma\><rsub|z>|i\<omega\>\<varepsilon\>><right|)>|1+<frac|\<sigma\><rsub|x>|i\<omega\>\<varepsilon\>>>>>|<row|<cell|>|<cell|=>|<cell|<frac|1+<frac|\<sigma\><rsub|z>+\<sigma\><rsub|y>|i\<omega\>\<varepsilon\>>+<frac|\<sigma\><rsub|y>\<sigma\><rsub|z>|(i\<omega\>\<varepsilon\>)<rsup|2>>|1+<frac|\<sigma\><rsub|x>|i\<omega\>\<varepsilon\>>>=<frac|i\<omega\>\<varepsilon\>(i\<omega\>\<varepsilon\>+\<sigma\><rsub|z>+\<sigma\><rsub|y>)+\<sigma\><rsub|z>\<sigma\><rsub|y>|(i\<omega\>\<varepsilon\>)<rsup|2>>\<cdot\><frac|i\<omega\>\<varepsilon\>|i\<omega\>\<varepsilon\>+\<sigma\><rsub|x>>>>|<row|<cell|>|<cell|=>|<cell|<frac|i\<omega\>\<varepsilon\>(i\<omega\>\<varepsilon\>+\<sigma\><rsub|z>+\<sigma\><rsub|y>)+\<sigma\><rsub|z>\<sigma\><rsub|y>|i\<omega\>\<varepsilon\>>\<cdot\><frac|1|i\<omega\>\<varepsilon\>+\<sigma\><rsub|x>>>>|<row|<cell|>|<cell|=>|<cell|<frac|i\<omega\>\<varepsilon\>+\<sigma\><rsub|z>+\<sigma\><rsub|y>|i\<omega\>\<varepsilon\>+\<sigma\><rsub|x>>+<frac|\<sigma\><rsub|z>\<sigma\><rsub|y>|i\<omega\>\<varepsilon\>(i\<omega\>\<varepsilon\>+\<sigma\><rsub|x>)>.>>>>
  </eqnarray*>

  To avoid keeping two state terms around, we split
  <with|mode|math|<wide|S|~>> as follows:

  <\equation*>
    <wide|S|~>=S<rsup|\<div\>>S<rsup|\<leftarrow\>>,<space|1em>S<rsup|\<div\>>\<assign\><matrix|<tformat|<table|<row|<cell|s<rsub|z>/s<rsub|x>>|<cell|>|<cell|>>|<row|<cell|>|<cell|s<rsub|x>/s<rsub|y>>|<cell|>>|<row|<cell|>|<cell|>|<cell|s<rsub|y>/s<rsub|z>>>>>>,<space|1em>S<rsup|\<leftarrow\>>\<assign\><matrix|<tformat|<table|<row|<cell|s<rsub|y>>|<cell|>|<cell|>>|<row|<cell|>|<cell|s<rsub|z>>|<cell|>>|<row|<cell|>|<cell|>|<cell|s<rsub|x>>>>>>.
  </equation*>

  Now write Maxwell's equations inside the PML:

  <\eqnarray*>
    <tformat|<table|<row|<cell|i\<omega\><matrix|<tformat|<table|<row|<cell|\<varepsilon\>>|<cell|>>|<row|<cell|>|<cell|-\<mu\><rsub|0>>>>>><matrix|<tformat|<table|<row|<cell|<wide|S|~>>>|<row|<cell|<wide|S|~>>>>>><matrix|<tformat|<table|<row|<cell|E>>|<row|<cell|H>>>>>>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|\<nabla\>\<times\>H>>|<row|<cell|\<nabla\>\<times\>E>>>>>>>>>
  </eqnarray*>

  We define, slightly arbitrarily,

  <\eqnarray*>
    <tformat|<table|<row|<cell|D>|<cell|\<assign\>>|<cell|\<varepsilon\><rsub|>S<rsup|\<div\>>E,>>|<row|<cell|B>|<cell|\<assign\>>|<cell|\<mu\><rsub|0>S<rsup|\<div\>>H,>>>>
  </eqnarray*>

  and obtain a simpler time-domain evolution of <with|mode|math|D> and
  <with|mode|math|B>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|i\<omega\>S<rsup|\<leftarrow\>><matrix|<tformat|<table|<row|<cell|D>>|<row|<cell|-B>>>>>>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|\<nabla\>\<times\>H>>|<row|<cell|\<nabla\>\<times\>E>>>>>>>|<row|<cell|\<Leftrightarrow\>i\<omega\>(i\<omega\>\<varepsilon\>+\<Sigma\><rsup|\<leftarrow\>>)<matrix|<tformat|<table|<row|<cell|D>>|<row|<cell|-B>>>>>>|<cell|=>|<cell|i\<omega\>\<varepsilon\><matrix|<tformat|<table|<row|<cell|\<nabla\>\<times\>H>>|<row|<cell|\<nabla\>\<times\>E>>>>>>>|<row|<cell|\<Leftrightarrow\>(i\<omega\>\<varepsilon\>+\<Sigma\><rsup|\<leftarrow\>>)<matrix|<tformat|<table|<row|<cell|D>>|<row|<cell|-B>>>>>>|<cell|=>|<cell|\<varepsilon\><rsub|0><matrix|<tformat|<table|<row|<cell|\<nabla\>\<times\>H>>|<row|<cell|\<nabla\>\<times\>E>>>>>>>|<row|<cell|\<Leftrightarrow\>\<partial\><rsub|t><matrix|<tformat|<table|<row|<cell|D>>|<row|<cell|B>>>>>>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|\<nabla\>\<times\>H>>|<row|<cell|-\<nabla\>\<times\>E>>>>>-<frac|\<Sigma\><rsup|\<leftarrow\>>|\<varepsilon\>><matrix|<tformat|<table|<row|<cell|D>>|<row|<cell|B>>>>>.>>>>
  </eqnarray*>

  where we let

  <\equation*>
    \<Sigma\><rsup|\<leftarrow\>>\<assign\><matrix|<tformat|<table|<row|<cell|\<sigma\><rsub|y>>|<cell|>|<cell|>>|<row|<cell|>|<cell|\<sigma\><rsub|z>>|<cell|>>|<row|<cell|>|<cell|>|<cell|\<sigma\><rsub|x>>>>>>.
  </equation*>

  Now, how do we obtain <with|mode|math|E> from <with|mode|math|D> and
  <with|mode|math|H> from <with|mode|math|B>? Let's tackle <with|mode|math|E>
  first:

  <\eqnarray*>
    <tformat|<table|<row|<cell|D>|<cell|=>|<cell|\<varepsilon\>S<rsup|\<div\>>E>>|<row|<cell|\<Leftrightarrow\>(i\<omega\>\<varepsilon\>+\<Sigma\>)D>|<cell|=>|<cell|\<varepsilon\>(i\<omega\>\<varepsilon\>+\<Sigma\><rsup|\<rightarrow\>>)E>>|<row|<cell|\<Leftrightarrow\>\<varepsilon\>\<partial\><rsub|t>D+\<Sigma\>D>|<cell|=>|<cell|\<varepsilon\><rsup|2>\<partial\><rsub|t>E+\<varepsilon\>\<Sigma\><rsup|\<rightarrow\>>E>>|<row|<cell|\<Leftrightarrow\>\<partial\><rsub|t>E>|<cell|=>|<cell|<frac|1|\<varepsilon\>>\<partial\><rsub|t>D+<frac|1|\<varepsilon\><rsup|2>>\<Sigma\>D-<frac|1|\<varepsilon\>>\<Sigma\><rsup|\<rightarrow\>>E>>|<row|<cell|\<Leftrightarrow\>\<partial\><rsub|t>E>|<cell|=>|<cell|<frac|1|\<varepsilon\>><left|[>\<nabla\>\<times\>H-<frac|\<Sigma\><rsup|\<leftarrow\>>|\<varepsilon\>>D+<frac|1|\<varepsilon\>>\<Sigma\>D-\<Sigma\><rsup|\<rightarrow\>>E<right|]>,>>>>
  </eqnarray*>

  with

  <\equation*>
    \<Sigma\>\<assign\><matrix|<tformat|<table|<row|<cell|\<sigma\><rsub|x>>|<cell|>|<cell|>>|<row|<cell|>|<cell|\<sigma\><rsub|y>>|<cell|>>|<row|<cell|>|<cell|>|<cell|\<sigma\><rsub|z>>>>>>,<space|1em>\<Sigma\><rsup|\<rightarrow\>>\<assign\><matrix|<tformat|<table|<row|<cell|\<sigma\><rsub|z>>|<cell|>|<cell|>>|<row|<cell|>|<cell|\<sigma\><rsub|x>>|<cell|>>|<row|<cell|>|<cell|>|<cell|\<sigma\><rsub|y>>>>>>.
  </equation*>

  Likewise, for <with|mode|math|H> we obtain

  <\eqnarray*>
    <tformat|<table|<row|<cell|B>|<cell|=>|<cell|\<mu\><rsub|0>S<rsup|\<div\>>H>>|<row|<cell|\<Leftrightarrow\>(i\<omega\>\<varepsilon\>+\<Sigma\>)B>|<cell|=>|<cell|\<mu\><rsub|0>(i\<omega\>\<varepsilon\>+\<Sigma\><rsup|\<rightarrow\>>)H>>|<row|<cell|\<Leftrightarrow\>\<varepsilon\>\<partial\><rsub|t>B+\<Sigma\>B>|<cell|=>|<cell|\<varepsilon\>\<mu\><rsub|0>\<partial\><rsub|t>H+\<mu\><rsub|0>\<Sigma\><rsup|\<rightarrow\>>H>>|<row|<cell|\<Leftrightarrow\>\<varepsilon\>\<mu\><rsub|0>\<partial\><rsub|t>H>|<cell|=>|<cell|\<varepsilon\>\<partial\><rsub|t>B+\<Sigma\>B-\<mu\><rsub|0>\<Sigma\><rsup|\<rightarrow\>>H>>|<row|<cell|\<Leftrightarrow\>\<partial\><rsub|t>H>|<cell|=>|<cell|<frac|1|\<mu\><rsub|0>>\<partial\><rsub|t>B+<frac|1|\<varepsilon\>\<mu\><rsub|0>>\<Sigma\>B-<frac|1|\<varepsilon\>>\<Sigma\><rsup|\<rightarrow\>>H>>|<row|<cell|\<Leftrightarrow\>\<partial\><rsub|t>H>|<cell|=>|<cell|<frac|1|\<mu\><rsub|0>><left|[>-\<nabla\>\<times\>E-<frac|\<Sigma\><rsup|\<leftarrow\>>|\<varepsilon\>>B<right|]>+<frac|1|\<varepsilon\>\<mu\><rsub|0>>\<Sigma\>B-<frac|1|\<varepsilon\>>\<Sigma\><rsup|\<rightarrow\>>H.>>>>
  </eqnarray*>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-type|letter>
    <associate|sfactor|4>
  </collection>
</initial>