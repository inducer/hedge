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
    </input>

    <\output>
      <with|mode|math|math-display|true|<frac|<left|(>s<rsup|2>-2s+1<right|)><sqrt|2>f<left|(><frac|-s-2r-1|s-1><right|)><left|(>-s+1<right|)><rsup|i>g<rsub|
      ><rsup|,><left|(>s<right|)>+<left|(>2r+2<right|)><sqrt|2>g<left|(>s<right|)><left|(>-s+1<right|)><rsup|i>f<rsub|
      ><rsup|,><left|(><frac|-s-2r-1|s-1><right|)>+<left|(>-i*s<rsup|2>+2i*s-i<right|)><sqrt|2>f<left|(><frac|-s-2r-1|s-1><right|)>g<left|(>s<right|)><left|(>-s+1<right|)><rsup|<left|(>i-1<right|)>>|s<rsup|2>-2s+1><leqno>(11)>

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

  \;
</body>

<\initial>
  <\collection>
    <associate|page-orientation|landscape>
    <associate|page-type|letter>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|2|?>>
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
    </associate>
  </collection>
</auxiliary>