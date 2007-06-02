<TeXmacs|1.0.6>

<style|<tuple|generic|maxima|axiom>>

<\body>
  <section|Mapping from 2D equilateral to Unit Coordinates>

  <with|prog-language|axiom|prog-session|default|<\session>
    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      )clear all
    </input>

    <\output>
      \ \ \ All user variables and function definitions have been cleared.
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      verteq:=[2/sqrt(3)*vector [sin(i*2*%pi/3),cos(i*2*%pi/3)] for i in
      0..2]
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[><left|[>0,<space|0.5spc><frac|2<sqrt|3>|3><right|]>,<space|0.5spc><left|[>1,<space|0.5spc>-<frac|<sqrt|3>|3><right|]>,<space|0.5spc><left|[>-1,<space|0.5spc>-<frac|<sqrt|3>|3><right|]><right|]><leqno>(1)>

      <axiomtype|List Vector Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      vertun:= [vector[-1,1], vector[1,-1], vector[-1,-1]]
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[><left|[>-1,<space|0.5spc>1<right|]>,<space|0.5spc><left|[>1,<space|0.5spc>-1<right|]>,<space|0.5spc><left|[>-1,<space|0.5spc>-1<right|]><right|]><leqno>(2)>

      <axiomtype|List Vector Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      A:=matrix [[a,b],[c,d]]
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[><tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|c>|<cwith|1|-1|2|2|cell-halign|c>|<table|<row|<cell|a>|<cell|b>>|<row|<cell|c>|<cell|d>>>>><right|]><leqno>(3)>

      <axiomtype|Matrix Polynomial Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      v:=vector [e,f]
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[>e,<space|0.5spc>f<right|]><leqno>(4)>

      <axiomtype|Vector OrderedVariableList [e,f] >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      eqns:=concat(

      [(A*verteq.i+v).1=vertun.i.1 for i in 1..3],

      [(A*verteq.i+v).2=vertun.i.2 for i in 1..3])
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[><frac|2b<sqrt|3>+3e|3>=-1,<space|0.5spc><frac|-b<sqrt|3>+3e+3a|3>=1,<space|0.5spc><frac|-b<sqrt|3>+3e-3a|3>=-1,<space|0.5spc><frac|2d<sqrt|3>+3f|3>=1,<space|0.5spc><frac|-d<sqrt|3>+3f+3c|3>=-1,<space|0.5spc><frac|-d<sqrt|3>+3f-3c|3>=-1<right|]><leqno>(5)>

      <axiomtype|List Equation Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      solve(eqns,[a,b,c,d,e,f])
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[><left|[>a=1,<space|0.5spc>b=-<frac|1|<sqrt|3>>,<space|0.5spc>c=0,<space|0.5spc>d=<frac|2|<sqrt|3>>,<space|0.5spc>e=-<frac|1|3>,<space|0.5spc>f=-<frac|1|3><right|]><right|]><leqno>(6)>

      <axiomtype|List List Equation Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      \;
    </input>
  </session>>

  <section|Derivative of the 2D Basis functions>

  <with|prog-language|axiom|prog-session|default|<\session>
    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      )clear all
    </input>

    <\output>
      \ \ \ All user variables and function definitions have been cleared.
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      f:=operator 'f
    </input>

    <\output>
      <with|mode|math|math-display|true|f<leqno>(1)>

      <axiomtype|BasicOperator >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      g:=operator 'g
    </input>

    <\output>
      <with|mode|math|math-display|true|g<leqno>(2)>

      <axiomtype|BasicOperator >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      a:=2*(1+r)/(1-s)-1
    </input>

    <\output>
      <with|mode|math|math-display|true|<frac|-s-2r-1|s-1><leqno>(3)>

      <axiomtype|Fraction Polynomial Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      D(a,r)
    </input>

    <\output>
      <with|mode|math|math-display|true|-<frac|2|s-1><leqno>(4)>

      <axiomtype|Fraction Polynomial Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      D(a,s)
    </input>

    <\output>
      <with|mode|math|math-display|true|<frac|2r+2|s<rsup|2>-2s+1><leqno>(5)>

      <axiomtype|Fraction Polynomial Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      expr:=sqrt(2)*f(a)*g(s)*(1-s)^i
    </input>

    <\output>
      <with|mode|math|math-display|true|<sqrt|2>f<left|(><frac|-s-2r-1|s-1><right|)>g<left|(>s<right|)><left|(>-s+1<right|)><rsup|i><leqno>(6)>

      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      D(expr,r)
    </input>

    <\output>
      <with|mode|math|math-display|true|-<frac|2<sqrt|2>g<left|(>s<right|)><left|(>-s+1<right|)><rsup|i>f<rsub|
      ><rsup|,><left|(><frac|-s-2r-1|s-1><right|)>|s-1><leqno>(7)>

      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      D(expr,s)

      *

      (s-1)^2
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|(>s<rsup|2>-2s+1<right|)><sqrt|2>f<left|(><frac|-s-2r-1|s-1><right|)><left|(>-s+1<right|)><rsup|i>g<rsub|
      ><rsup|,><left|(>s<right|)>+<left|(>2r+2<right|)><sqrt|2>g<left|(>s<right|)><left|(>-s+1<right|)><rsup|i>f<rsub|
      ><rsup|,><left|(><frac|-s-2r-1|s-1><right|)>+<left|(>-i*s<rsup|2>+2i*s-i<right|)><sqrt|2>f<left|(><frac|-s-2r-1|s-1><right|)>g<left|(>s<right|)><left|(>-s+1<right|)><rsup|<left|(>i-1<right|)>><leqno>(9)>

      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      \;
    </input>
  </session>>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|>|<cell|<frac|1|(s-1)<rsup|2>><left|[><left|(>s<rsup|2>-2s+1<right|)><sqrt|2>f<left|(><frac|-s-2r-1|s-1><right|)><left|(>-s+1<right|)><rsup|i>g<rsub|
    ><rsup|,><left|(>s<right|)><right|]>>>|<row|<cell|>|<cell|>|<cell|+<frac|1|(s-1)<rsup|2>><left|[><left|(>2r+2<right|)><sqrt|2>g<left|(>s<right|)><left|(>-s+1<right|)><rsup|i>f<rsub|
    ><rsup|,><left|(><frac|-s-2r-1|s-1><right|)><right|]>>>|<row|<cell|>|<cell|>|<cell|+<frac|1|(s-1)<rsup|2>><left|[><left|(>-i*s<rsup|2>+2i*s-i<right|)><sqrt|2>f<left|(><frac|-s-2r-1|s-1><right|)>g<left|(>s<right|)><left|(>-s+1<right|)><rsup|<left|(>i-1<right|)>><right|]>>>|<row|<cell|>|<cell|=>|<cell|<sqrt|2><left|[>f<left|(>a<right|)><left|(>1-s<right|)><rsup|i>g<rsub|
    ><rsup|,><left|(>s<right|)>>>|<row|<cell|>|<cell|>|<cell|+<left|(>2r+2<right|)>g<left|(>s<right|)><left|(>1-s<right|)><rsup|i-2>f<rsub|
    ><rsup|,><left|(>a<right|)>>>|<row|<cell|>|<cell|>|<cell|-i*f<left|(>a<right|)>g<left|(>s<right|)><left|(>1-s<right|)><rsup|<left|(>i-1<right|)>><right|]>>>>>
  </eqnarray*>

  <section|Weak DG for the Wave Equation>

  We have <with|mode|math|u<rsub|t t>-\<Delta\>u=0>, so

  <\eqnarray*>
    <tformat|<table|<row|<cell|u<rsub|t>-\<nabla\>\<cdot\>v>|<cell|=>|<cell|0,>>|<row|<cell|v<rsub|t>-\<nabla\>u>|<cell|=>|<cell|0.>>>>
  </eqnarray*>

  Note that <with|mode|math|v> is a vector. Let <with|mode|math|l<rsub|i>> be
  the <with|mode|math|i>th Lagrange interpolation polynomial and
  <with|mode|math|\<b-u\>=(u<rsub|i>)> be the vector of nodal coefficients of
  our approximated function <with|mode|math|<wide|u|~>(x)=u<rsub|i>l<rsub|i>(x)>.
  (<with|mode|math|u> without subscript denotes a continuous function
  <with|mode|math|u>.) Then, using the summation convention,

  <\eqnarray*>
    <tformat|<table|<row|<cell|0>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|T<rsub|k>>(\<nabla\>\<cdot\>v)\<varphi\>>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|D<rsub|k>>v\<cdot\>\<nabla\>\<varphi\>-<big|int><rsub|T<rsub|k>>\<nabla\>\<cdot\>(v\<varphi\>)>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|D<rsub|k>>\<b-v\>\<cdot\>\<nabla\>\<varphi\>-<big|int><rsub|\<partial\>T<rsub|k>>v\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|\<approx\>>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|D<rsub|k>>\<b-v\>\<cdot\>\<nabla\>\<varphi\>-<big|int><rsub|\<partial\>T<rsub|k>>v<rsup|\<ast\>>\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|\<approx\>>|<cell|<big|int><rsub|T<rsub|k>>\<partial\><rsub|t>u<rsub|i>l<rsub|i>l<rsub|j>+<big|int><rsub|T<rsub|k>><matrix|<tformat|<table|<row|<cell|[v<rsub|i>]<rsub|1>l<rsub|i>>>|<row|<cell|[v<rsub|i>]<rsub|2>l<rsub|i>>>>>>\<cdot\><matrix|<tformat|<table|<row|<cell|\<partial\><rsub|x<rsub|2>>l<rsub|j>>>|<row|<cell|\<partial\><rsub|x<rsub|2>>l<rsub|j>>>>>>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><big|int><rsub|F><matrix|<tformat|<table|<row|<cell|[v<rsub|i>]<rsup|\<ast\>><rsub|1>l<rsub|i><rsup|F>l<rsub|j><rsup|F>>>|<row|<cell|[v<rsub|i>]<rsup|\<ast\>><rsub|2>l<rsub|i><rsup|F>l<rsub|i><rsup|F>>>>>>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|M<rsub|i
    j><rsup|k>\<partial\><rsub|t>u<rsub|i>+S<rsub|i
    j,\<partial\>x<rsub|1>><rsup|k>[v<rsub|i>]<rsub|1>+S<rsub|i
    j,\<partial\>x<rsub|2>><rsup|k>[v<rsub|i>]<rsub|2>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><left|[>[v<rsub|i>]<rsup|\<ast\>><rsub|1>M<rsub|i
    j><rsup|F>n<rsub|1>+[v<rsub|i>]<rsup|\<ast\>><rsub|2>M<rsub|i
    j><rsup|F>n<rsub|2><right|]>>>|<row|<cell|>|<cell|=>|<cell|(M<rsup|k>)<rsup|T>\<partial\><rsub|t>\<b-u\>+(S<rsub|\<partial\>x<rsub|1>><rsup|k>)<rsup|T>\<b-v\><rsub|1>+(S<rsub|\<partial\>x<rsub|2>><rsup|k>)<rsup|T>\<b-v\><rsub|2>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><left|[>[v<rsub|F>]<rsup|\<ast\>><rsub|1>(M<rsup|F>)<rsup|T>n<rsub|1>+[v<rsub|F>]<rsup|\<ast\>><rsub|2>(M<rsup|F>)<rsup|T>n<rsub|2><right|]>>>>>
  </eqnarray*>

  where the <with|mode|math|F\<subset\>\<partial\>T<rsub|k>> represent the
  discrete faces of the element <with|mode|math|T<rsub|k>>, and
  <with|mode|math|l<rsub|i><rsup|F>> is the Lagrange interpolation polynomial
  on the face <with|mode|math|f> corresponding to the node with global number
  <with|mode|math|i>, or constant zero if there is no such function (such as
  when <with|mode|math|i> is not the number of a node in <with|mode|math|F>).
  Further,

  <\eqnarray*>
    <tformat|<table|<row|<cell|M<rsub|i j><rsup|k>>|<cell|\<assign\>>|<cell|<big|int><rsub|T<rsub|k>>l<rsub|i>l<rsub|j>,>>|<row|<cell|S<rsub|i
    j,\<partial\>x<rsub|\<nu\>>><rsup|k>>|<cell|\<assign\>>|<cell|<big|int><rsub|T<rsub|k>>l<rsub|i>\<partial\><rsub|x<rsub|\<nu\>>>l<rsub|j>,>>|<row|<cell|M<rsub|i
    j><rsup|F>>|<cell|\<assign\>>|<cell|<big|int><rsub|F>l<rsub|i><rsup|F>l<rsub|j><rsup|F>.>>>>
  </eqnarray*>

  Note that <with|mode|math|M<rsup|k>=(M<rsup|k>)<rsup|T>> and
  <with|mode|math|M<rsup|F>=(M<rsup|F>)<rsup|T>>.

  Recall <with|mode|math|V<wide|\<b-u\>|^>=\<b-u\>> where
  <with|mode|math|<wide|u|^><rsub|i>p<rsub|i>=u<rsub|i>l<rsub|i>>, and

  <\equation*>
    V<rsup|T>\<b-l\>(x)=<matrix|<tformat|<table|<row|<cell|p<rsub|1>(x<rsub|1>)>|<cell|>|<cell|p(x<rsub|n>)>>|<row|<cell|\<vdots\>>|<cell|>|<cell|\<vdots\>>>|<row|<cell|p<rsub|n>(x<rsub|1>)>|<cell|\<cdots\>>|<cell|p<rsub|n>(x<rsub|n>)>>>>>\<b-l\>(x)=<matrix|<tformat|<table|<row|<cell|l<rsub|1>(x)p<rsub|1>(x<rsub|1>)+\<cdots\>+l<rsub|n>(x)p<rsub|1>(x<rsub|n>)>>|<row|<cell|\<vdots\>>>|<row|<cell|l<rsub|1>(x)p<rsub|n>(x<rsub|1>)+\<cdots\>+l<rsub|n>(x)p<rsub|n>(x<rsub|n>)>>>>>=<matrix|<tformat|<table|<row|<cell|p<rsub|1>(x)>>|<row|<cell|\<vdots\>>>|<row|<cell|p<rsub|n>(x)>>>>>,
  </equation*>

  simply because the sum of Lagrange polynomials uniquely interpolates
  <with|mode|math|p<rsub|i>(x)>. We thus find

  <\eqnarray*>
    <tformat|<table|<row|<cell|M<rsup|k><rsub|i
    j>>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>l<rsub|i>l<rsub|j>=J<rsub|k><big|int><rsub|T>([V<rsup|-T>]<rsub|i
    m>p<rsub|m>)([V<rsup|-T>]<rsub|j n>p<rsub|n>)>>|<row|<cell|>|<cell|=>|<cell|J<rsub|k>[V<rsup|-T>]<rsub|i
    m>[V<rsup|-T>]<rsub|j n><big|int><rsub|T>p<rsub|m>p<rsub|n>>>|<row|<cell|>|<cell|=>|<cell|J<rsub|k>[V<rsup|-T>]<rsub|i
    m>[V<rsup|-T>]<rsub|j n>\<delta\><rsub|m,n>>>|<row|<cell|>|<cell|=>|<cell|J<rsub|k>[V<rsup|-T>]<rsub|i
    n>[V<rsup|-T>]<rsub|j n>>>|<row|<cell|>|<cell|=>|<cell|J<rsub|k>[V<rsup|-T>]<rsub|i
    n>[V<rsup|-1>]<rsub|n j>>>|<row|<cell|>|<cell|=>|<cell|J<rsub|k>[(V*V<rsup|T>)<rsup|-1>]<rsub|i
    j>.>>>>
  </eqnarray*>

  To obtain the stiffness matrix <with|mode|math|S<rsub|i
  j,\<partial\>x<rsub|\<nu\>>>>, realize that we really only need a
  differentiation matrix <with|mode|math|D<rsub|x<rsub|\<nu\>>>> that maps
  the nodal values of a function to those of its derivative. If we have
  <with|mode|math|D<rsub|x<rsub|\<nu\>>>>, we can simply recycle the mass
  matrix:

  <\equation*>
    S<rsub|i,j,\<partial\>x<rsub|\<nu\>>>=M*D<rsub|x<rsub|\<nu\>>>.
  </equation*>

  Before finding the global differentiation matrix
  <with|mode|math|D<rsub|x<rsub|\<nu\>>>>, we first endeavor to find the
  local differentiation matrix <with|mode|math|D<rsub|r<rsub|\<nu\>>>>, where
  we denote local (unit) coordinates <with|mode|math|r> and global
  coordinates <with|mode|math|x>. We define the derivative Vandermonde
  matrices <with|mode|math|V<rsub|\<partial\>r<rsub|\<nu\>>>\<assign\>[\<partial\><rsub|\<nu\>>p<rsub|j>(r<rsub|i>)]<rsub|i
  j><rsub|>>. Recalling <with|mode|math|V<wide|\<b-u\>|^>=\<b-u\>>, we obtain
  <with|mode|math|V<rsub|\<partial\>r<rsub|\<nu\>>><wide|\<b-u\>|^>=V<rsub|\<partial\>r<rsub|\<nu\>>>V<rsup|-1>\<b-u\>=D<rsub|r<rsub|\<nu\>>>\<b-u\>>
  and thus <with|mode|math|D<rsub|r<rsub|\<nu\>>>=V<rsub|\<partial\>r<rsub|\<nu\>>>V<rsup|-1>>.
  For clarity, and again using the summation convention, consider the
  identity

  <\equation*>
    <wide|u|~><rprime|'>(x)=u<rsub|j>l<rsub|j><rprime|'>(x)=u<rsub|j><wide*|l<rsub|j><rprime|'>(x<rsup|(i)>)|\<wide-underbrace\>><rsub|D<rsub|i,j>>l<rsub|i>(x).
  </equation*>

  If we define a function <with|mode|math|F<rsub|k>:r\<mapsto\>x>, then the
  global differentiation matrix is then given by

  <\equation*>
    [D<rsub|x<rsub|\<nu\>>>]<rsub|i,j>=\<partial\><rsub|x<rsub|\<nu\>>>(l<rsub|j>(F<rsup|-1><rsub|k>(x<rsup|(i)>)))=(\<partial\><rsub|r>l<rsub|j>)(r<rsup|(i)><rsup|>)<left|[>(F<rsub|k><rsup|-1>)<rprime|'>(x<rsup|(i)>)<right|]><rsub|\<mu\>,\<nu\>>=[D<rsub|r<rsub|\<mu\>>>]<rsub|i,j><left|[>(F<rsup|-1><rsub|k>)<rprime|'>(x<rsup|(i)>)<right|]><rsub|\<mu\>,\<nu\>>.
  </equation*>

  We set <with|mode|math|\<b-v\><rsup|\<ast\>>\<assign\>{\<b-v\>}>. Likewise,
  we get

  <\eqnarray*>
    <tformat|<table|<row|<cell|0>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>v<rsub|t>\<cdot\>\<psi\>-<big|int><rsub|T<rsub|k>>\<nabla\>u\<cdot\>\<psi\>>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>v<rsub|t>\<cdot\>\<psi\>+<big|int><rsub|T<rsub|k>>u\<nabla\>\<cdot\>\<psi\>-<big|int><rsub|T<rsub|k>>\<nabla\>\<cdot\>(u<with|math-font-series|bold|\<psi\>>)>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>v<rsub|t>\<cdot\>\<psi\>+<big|int><rsub|T<rsub|k>>u\<nabla\>\<cdot\>\<psi\>-<big|int><rsub|\<partial\>T<rsub|k>>u\<psi\>\<cdot\>n>>|<row|<cell|>|<cell|\<approx\>>|<cell|<big|int><rsub|T<rsub|k>>v<rsub|t>\<cdot\>\<psi\>+<big|int><rsub|T<rsub|k>>u\<nabla\>\<cdot\>\<psi\>-<big|int><rsub|\<partial\>T<rsub|k>>u<rsup|\<ast\>>\<psi\>\<cdot\>n>>|<row|<cell|>|<cell|\<approx\>>|<cell|<choice|<tformat|<table|<row|<cell|<with|math-display|true|<big|int><rsub|T<rsub|k>><matrix|<tformat|<table|<row|<cell|\<partial\><rsub|t>[v<rsub|i>]<rsub|1>l<rsub|i>>>|<row|<cell|\<partial\><rsub|t>[v<rsub|i>]<rsub|2>l<rsub|i>>>>>>\<cdot\><matrix|<tformat|<table|<row|<cell|l<rsub|j>>>|<row|<cell|0>>>>>+<big|int><rsub|T<rsub|k>>u<rsub|i>l<rsub|i>\<nabla\>\<cdot\><matrix|<tformat|<table|<row|<cell|l<rsub|j>>>|<row|<cell|0>>>>>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><big|int><rsub|F><matrix|<tformat|<table|<row|<cell|u<rsub|i><rsup|\<ast\>>l<rsub|i><rsup|F>l<rsub|j><rsup|F>>>|<row|<cell|u<rsub|i><rsup|\<ast\>>l<rsub|i><rsup|F>0>>>>>\<cdot\>n>>>|<row|<cell|<with|math-display|true|<big|int><rsub|T<rsub|k>><matrix|<tformat|<table|<row|<cell|\<partial\><rsub|t>[v<rsub|i>]<rsub|1>l<rsub|i>>>|<row|<cell|\<partial\><rsub|t>[v<rsub|i>]<rsub|2>l<rsub|i>>>>>>\<cdot\><matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|l<rsub|j>>>>>>+<big|int><rsub|T<rsub|k>>u<rsub|i>l<rsub|i>\<nabla\>\<cdot\><matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|l<rsub|j>>>>>>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><big|int><rsub|F><matrix|<tformat|<table|<row|<cell|u<rsub|i><rsup|\<ast\>>l<rsub|i><rsup|F>0>>|<row|<cell|u<rsub|i><rsup|\<ast\>>l<rsub|i><rsup|F>l<rsub|j><rsup|F>>>>>>\<cdot\>n>>>>>>>>|<row|<cell|>|<cell|=>|<cell|<choice|<tformat|<table|<row|<cell|<with|math-display|true|<big|int><rsub|T<rsub|k>>\<partial\><rsub|t>[v<rsub|i>]<rsub|1>l<rsub|i>l<rsub|j>+<big|int><rsub|T<rsub|k>>u<rsub|i>l<rsub|i>\<partial\><rsub|x<rsub|1>>l<rsub|j>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><big|int><rsub|F>u<rsub|i><rsup|\<ast\>>l<rsub|i><rsup|F>l<rsub|j><rsup|F>n<rsub|x>>>>|<row|<cell|<with|math-display|true|<big|int><rsub|T<rsub|k>>\<partial\><rsub|t>[v<rsub|i>]<rsub|2>l<rsub|i>l<rsub|j>+<big|int><rsub|T<rsub|k>>u<rsub|i>l<rsub|i>\<partial\><rsub|x<rsub|2>>l<rsub|j>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><big|int><rsub|F>u<rsub|i><rsup|\<ast\>>l<rsub|i><rsup|F>l<rsub|j><rsup|F>n<rsub|y>>>>>>>>>|<row|<cell|>|<cell|=>|<cell|<choice|<tformat|<table|<row|<cell|<with|math-display|true|\<partial\><rsub|t>[v<rsub|i>]<rsub|1>M<rsub|i
    j><rsup|k>+u<rsub|i>S<rsup|k><rsub|i j,\<partial\>x<rsub|2>>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>>u<rsub|i><rsup|\<ast\>>M<rsub|i
    j><rsup|F>n<rsub|x>>>>|<row|<cell|<with|math-display|true|\<partial\><rsub|t>[v<rsub|i>]<rsub|2>M<rsub|i
    j><rsup|k>+u<rsub|i>S<rsup|k><rsub|i j,\<partial\>x<rsub|2>>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>>u<rsub|i><rsup|\<ast\>>M<rsub|i
    j><rsup|F>n<rsub|y>>>>>>>>>>>
  </eqnarray*>

  and we set <with|mode|math|u<rsup|\<ast\>>\<assign\>{u}>.
</body>

<\initial>
  <\collection>
    <associate|page-orientation|portrait>
    <associate|page-type|letter>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-3|<tuple|3|3>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Mapping
      from 2D equilateral to Unit Coordinates>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Derivative
      of the 2D Basis functions> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Weak
      DG for the Wave Equation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>