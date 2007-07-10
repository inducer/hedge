<TeXmacs|1.0.6>

<style|<tuple|generic|maxima|axiom>>

<\body>
  <section|Cylindrical TM Maxwell Cavity Mode>

  <with|prog-language|axiom|prog-session|default|<\session>
    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      )clear all
    </input>

    <\output>
      \ \ \ All user variables and function definitions have been cleared.
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      )library )dir "/home/andreas/axiom"
    </input>

    <\output>
      \ \ \ TexFormat is already explicitly exposed in frame initial\ 

      \ \ \ TexFormat will be automatically loaded when needed from\ 

      \ \ \ \ \ \ /home/andreas/axiom/TEX.NRLIB/code

      \ \ \ TexFormat1 is already explicitly exposed in frame initial\ 

      \ \ \ TexFormat1 will be automatically loaded when needed from\ 

      \ \ \ \ \ \ /home/andreas/axiom/TEX1.NRLIB/code
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      J:=operator 'J
    </input>

    <\output>
      <with|mode|math|math-display|true|J<leqno>(1)>

      <axiomtype|BasicOperator >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      psi(rho,phi) == J(gamma*rho)*exp(PP*%i*m*phi)
    </input>

    <\output>
      <axiomtype|Void >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      psiglob:=psi(sqrt(x^2+y^2),atan(y/x))
    </input>

    <\output>
      \ \ \ Compiling function psi with type (Expression Integer,Expression\ 

      \ \ \ \ \ \ Integer) -\<gtr\> Expression Complex Integer\ 

      <with|mode|math|math-display|true|J<left|(>\<gamma\><sqrt|y<rsup|2>+x<rsup|2>><right|)>e<rsup|<left|(>i*P*P*m*arctan
      <left|(><frac|y|x><right|)><right|)>><leqno>(3)>

      <axiomtype|Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      D(psi(rho,phi),rho)
    </input>

    <\output>
      \ \ \ Compiling function psi with type (Variable rho,Variable phi)
      -\<gtr\>\ 

      \ \ \ \ \ \ Expression Complex Integer\ 

      <with|mode|math|math-display|true|\<gamma\>e<rsup|<left|(>i*P*P*m\<phi\><right|)>>J<rsub|
      ><rsup|,><left|(>\<gamma\>\<rho\><right|)><leqno>(5)>

      <axiomtype|Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      cross(vector [0,0,1], vector [x,y,0])
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[>-y,<space|0.5spc>x,<space|0.5spc>0<right|]><leqno>(7)>

      <axiomtype|Vector Polynomial Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      \;
    </input>
  </session>>

  <section|Rectangular Cavity Mode>

  According to Jackson, p. 357, (8.17), we need to solve the Helmholtz
  equation

  <\eqnarray*>
    <tformat|<cwith|1|1|3|3|cell-halign|l>|<table|<row|<cell|(\<nabla\><rsup|2>+\<mu\>\<varepsilon\>\<omega\><rsup|2>)<matrix|<tformat|<table|<row|<cell|\<b-E\>>>|<row|<cell|\<b-B\>>>>>>>|<cell|=>|<cell|\<b-0\>,>>>>
  </eqnarray*>

  subject to <with|mode|math|n\<times\>\<b-E\>=0> and
  <with|mode|math|n\<cdot\>\<b-B\>=0>. The ansatz is

  <\equation*>
    \<b-E\>=<matrix|<tformat|<table|<row|<cell|E<rsub|x,x>(x)E<rsub|x,y>(y)E<rsub|x,z>(z)>>|<row|<cell|E<rsub|y,x>(x)E<rsub|y,y>(y)E<rsub|y,z>(z)>>|<row|<cell|E<rsub|z,x>(x)E<rsub|z,y>(y)E<rsub|z,z>(z)>>>>>
  </equation*>

  and likewise for <with|mode|math|\<b-B\>>. The boundary conditions are

  <\eqnarray*>
    <tformat|<table|<row|<cell|E<rsub|x>(x,<with|math-level|1|<tabular|<tformat|<table|<row|<cell|0>>|<row|<cell|b>>>>>>,z)>|<cell|=>|<cell|0,>>|<row|<cell|E<rsub|x>(x,y,<with|math-level|1|<tabular|<tformat|<table|<row|<cell|0>>|<row|<cell|c>>>>>>)>|<cell|=>|<cell|0,>>>>
  </eqnarray*>

  and so on, as well as

  <\eqnarray*>
    <tformat|<table|<row|<cell|H<rsub|x>(<with|math-level|1|<tabular|<tformat|<table|<row|<cell|0>>|<row|<cell|a>>>>>>,y,z)>|<cell|=>|<cell|0.>>>>
  </eqnarray*>

  So

  <\equation*>
    E<rsub|x>=\<alpha\><rsub|x>exp(i\<beta\><rsub|x>x)sin<left|(><frac|n\<pi\>y|b><right|)>sin<left|(><frac|o\<pi\>z|c><right|)>exp(-i\<omega\>t)=\<alpha\><rsub|x>e<rsub|x>s<rsub|y>s<rsub|z>
  </equation*>

  and analogous terms for <with|mode|math|E<rsub|y>> and
  <with|mode|math|E<rsub|z>> satisfy the first batch of boundary conditions.
  Because of the Helmholtz equation, we find that
  <with|mode|math|\<beta\><rsub|x>=m\<pi\>/a>; otherwise, not all vector
  components would share the same eigenvalue, which would not solve the
  equation.

  <with|prog-language|axiom|prog-session|default|<\session>
    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      )clear all
    </input>

    <\output>
      \ \ \ All user variables and function definitions have been cleared.
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      )library )dir "/home/andreas/axiom"
    </input>

    <\output>
      \ \ \ TexFormat is already explicitly exposed in frame initial\ 

      \ \ \ TexFormat will be automatically loaded when needed from\ 

      \ \ \ \ \ \ /home/andreas/axiom/TEX.NRLIB/code

      \ \ \ TexFormat1 is already explicitly exposed in frame initial\ 

      \ \ \ TexFormat1 will be automatically loaded when needed from\ 

      \ \ \ \ \ \ /home/andreas/axiom/TEX1.NRLIB/code
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      factors:=[f,g,h];
    </input>

    <\output>
      <axiomtype|List OrderedVariableList [f,g,h] >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      coord := [x,y,z];
    </input>

    <\output>
      <axiomtype|List OrderedVariableList [x,y,z] >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      curl(v)== vector [D(v.3,y)-D(v.2,z),D(v.1,z)-D(v.3,x),D(v.2,x)-D(v.1,y)];
    </input>

    <\output>
      <axiomtype|Void >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      c:=1/sqrt(epsilon*mu);
    </input>

    <\output>
      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      sines(i) == sin(factors.i*coord.i);
    </input>

    <\output>
      <axiomtype|Void >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      cosines(i) == cos(factors.i*coord.i);
    </input>

    <\output>
      <axiomtype|Void >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      k:=sqrt(f^2+g^2+h^2);omega:=k*c;
    </input>

    <\output>
      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      zdep1:=exp(%i*h*z); zdep2:=exp(-%i*h*z);
    </input>

    <\output>
      <axiomtype|Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      z1f:=1; z2f:=-1;
    </input>

    <\output>
      <axiomtype|NonNegativeInteger >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      zdep:=z1f*zdep1 + z2f*zdep2;
    </input>

    <\output>
      <axiomtype|Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      C:=%i/(f^2+g^2);
    </input>

    <\output>
      <axiomtype|Fraction Polynomial Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      efield := vector [

      C*f*h*cosines(1)* \ sines(2)*zdep1,

      C*g*h* \ sines(1)*cosines(2)*zdep1,

      \ \ \ \ \ \ \ \ sines(1)* \ sines(2)*zdep];
    </input>

    <\output>
      \ \ \ Compiling function cosines with type PositiveInteger -\<gtr\>
      Expression\ 

      \ \ \ \ \ \ Integer\ 

      \ \ \ Compiling function sines with type PositiveInteger -\<gtr\>
      Expression\ 

      \ \ \ \ \ \ Integer\ 

      <axiomtype|Vector Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      hfield:=1/(-%i*omega*mu)*(-curl efield);
    </input>

    <\output>
      \ \ \ Compiling function curl with type Vector Expression Complex
      Integer

      \ \ \ \ \ \ \ -\<gtr\> Vector Expression Complex Integer\ 

      <axiomtype|Vector Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      efield2:=1/(-%i*omega*epsilon)*(curl hfield);
    </input>

    <\output>
      <axiomtype|Vector Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      efield2-efield
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[>0,<space|0.5spc>0,<space|0.5spc>0<right|]><leqno>(15)>

      <axiomtype|Vector Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      hfield
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[><frac|<left|(>-i*g*h<rsup|2>-i*g<rsup|3>-i*f<rsup|2>g<right|)>cos
      <left|(>g*y<right|)>e<rsup|<left|(>i*h*z<right|)>>sin
      <left|(>f*x<right|)><sqrt|\<epsilon\>\<mu\>>|<left|(>g<rsup|2>+f<rsup|2><right|)>\<mu\><sqrt|h<rsup|2>+g<rsup|2>+f<rsup|2>>>,<space|0.5spc><frac|<left|(>i*f*h<rsup|2>+i*f*g<rsup|2>+i*f<rsup|3><right|)>cos
      <left|(>f*x<right|)>e<rsup|<left|(>i*h*z<right|)>>sin
      <left|(>g*y<right|)><sqrt|\<epsilon\>\<mu\>>|<left|(>g<rsup|2>+f<rsup|2><right|)>\<mu\><sqrt|h<rsup|2>+g<rsup|2>+f<rsup|2>>>,<space|0.5spc>0<right|]><leqno>(71)>

      <axiomtype|Vector Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      hfield2:=vector [

      -%i*g/(f^2+g^2)*epsilon*omega*sines(1)*cosines(2)*exps(3),

      \ %i*f/(f^2+g^2)*epsilon*omega*cosines(1)*sines(2)*exps(3),

      0]
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[>-<frac|i\<epsilon\>g*cos
      <left|(>g*y<right|)>e<rsup|<left|(>i*h*z<right|)>>sin
      <left|(>f*x<right|)><sqrt|h<rsup|2>+g<rsup|2>+f<rsup|2>>|<left|(>g<rsup|2>+f<rsup|2><right|)><sqrt|\<epsilon\>\<mu\>>>,<space|0.5spc><frac|i\<epsilon\>f*cos
      <left|(>f*x<right|)>e<rsup|<left|(>i*h*z<right|)>>sin
      <left|(>g*y<right|)><sqrt|h<rsup|2>+g<rsup|2>+f<rsup|2>>|<left|(>g<rsup|2>+f<rsup|2><right|)><sqrt|\<epsilon\>\<mu\>>>,<space|0.5spc>0<right|]><leqno>(72)>

      <axiomtype|Vector Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      hfield-hfield2
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[>0,<space|0.5spc>0,<space|0.5spc>0<right|]><leqno>(73)>

      <axiomtype|Vector Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      bcs:=[

      eval(efield.3, x=0),

      eval(efield.3, y=0),

      eval(efield.3, z=0),

      eval(hfield.1, x=0),

      eval(hfield.2, y=0),

      eval(hfield.3, z=0)

      ]
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[>0,<space|0.5spc>0,<space|0.5spc>0,<space|0.5spc>0,<space|0.5spc>0,<space|0.5spc>0<right|]><leqno>(46)>

      <axiomtype|List Expression Complex Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      \;
    </input>
  </session>>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-type|letter>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-3|<tuple|3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Cylindrical
      TM Maxwell Cavity Mode> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Rectangular
      Cavity Mode> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>