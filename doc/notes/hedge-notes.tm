<TeXmacs|1.0.7>

<style|<tuple|article|maxima|axiom|mystyle>>

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

  <section|Mapping from 3D equilateral to Unit Coordinates>

  <with|prog-language|axiom|prog-session|default|<\session>
    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      )clear all
    </input>

    <\output>
      \ \ \ All user variables and function definitions have been cleared.
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      verteq:=[

      vector [-1,-1/sqrt(3),-1/sqrt(6)],

      vector [ 1,-1/sqrt(3),-1/sqrt(6)],

      vector [ 0, 2/sqrt(3),-1/sqrt(6)],

      vector [ 0, \ \ \ \ \ \ \ \ 0, 3/sqrt(6)]

      ]
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[><left|[>-1,<space|0.5spc>-<frac|<sqrt|3>|3>,<space|0.5spc>-<frac|<sqrt|6>|6><right|]>,<space|0.5spc><left|[>1,<space|0.5spc>-<frac|<sqrt|3>|3>,<space|0.5spc>-<frac|<sqrt|6>|6><right|]>,<space|0.5spc><left|[>0,<space|0.5spc><frac|2<sqrt|3>|3>,<space|0.5spc>-<frac|<sqrt|6>|6><right|]>,<space|0.5spc><left|[>0,<space|0.5spc>0,<space|0.5spc><frac|<sqrt|6>|2><right|]><right|]><leqno>(1)>

      <axiomtype|List Vector AlgebraicNumber >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      vertun:= [

      [-1,-1,-1],

      [ 1,-1,-1],

      [-1, 1,-1],

      [-1,-1, 1]

      ]
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[><left|[>-1,<space|0.5spc>-1,<space|0.5spc>-1<right|]>,<space|0.5spc><left|[>1,<space|0.5spc>-1,<space|0.5spc>-1<right|]>,<space|0.5spc><left|[>-1,<space|0.5spc>1,<space|0.5spc>-1<right|]>,<space|0.5spc><left|[>-1,<space|0.5spc>-1,<space|0.5spc>1<right|]><right|]><leqno>(2)>

      <axiomtype|List List Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      A:=matrix [[a,b,c],[d,e,f],[g,h,i]]
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[><tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|c>|<cwith|1|-1|2|2|cell-halign|c>|<cwith|1|-1|3|3|cell-halign|c>|<table|<row|<cell|a>|<cell|b>|<cell|c>>|<row|<cell|d>|<cell|e>|<cell|f>>|<row|<cell|g>|<cell|h>|<cell|i>>>>><right|]><leqno>(3)>

      <axiomtype|Matrix Polynomial Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      v:=vector [j,k,l]
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[>j,<space|0.5spc>k,<space|0.5spc>l<right|]><leqno>(4)>

      <axiomtype|Vector OrderedVariableList [j,k,l] >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      eqns:=concat(

      [(A*verteq.i+v).1=vertun.i.1 for i in 1..4],

      concat(

      [(A*verteq.i+v).2=vertun.i.2 for i in 1..4],

      [(A*verteq.i+v).3=vertun.i.3 for i in 1..4]

      ))
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[>j-<frac|<sqrt|6>|6>c-<frac|<sqrt|3>|3>b-a=-1,<space|0.5spc>j-<frac|<sqrt|6>|6>c-<frac|<sqrt|3>|3>b+a=1,<space|0.5spc>j-<frac|<sqrt|6>|6>c+<frac|2<sqrt|3>|3>b=-1,<space|0.5spc>j+<frac|<sqrt|6>|2>c=-1,<space|0.5spc>k-<frac|<sqrt|6>|6>f-<frac|<sqrt|3>|3>e-d=-1,<space|0.5spc>k-<frac|<sqrt|6>|6>f-<frac|<sqrt|3>|3>e+d=-1,<space|0.5spc>k-<frac|<sqrt|6>|6>f+<frac|2<sqrt|3>|3>e=1,<space|0.5spc>k+<frac|<sqrt|6>|2>f=-1,<space|0.5spc>l-<frac|<sqrt|6>|6>i-<frac|<sqrt|3>|3>h-g=-1,<space|0.5spc>l-<frac|<sqrt|6>|6>i-<frac|<sqrt|3>|3>h+g=-1,<space|0.5spc>l-<frac|<sqrt|6>|6>i+<frac|2<sqrt|3>|3>h=-1,<space|0.5spc>l+<frac|<sqrt|6>|2>i=1<right|]><leqno>(5)>

      <axiomtype|List Equation Polynomial AlgebraicNumber >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      solve(eqns,[a,b,c,d,e,f,g,h,i,j,k,l])
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|[><left|[>a=1,<space|0.5spc>b=-<frac|<sqrt|3>|3>,<space|0.5spc>c=-<frac|<sqrt|6>|6>,<space|0.5spc>d=0,<space|0.5spc>e=<frac|2<sqrt|3>|3>,<space|0.5spc>f=-<frac|<sqrt|6>|6>,<space|0.5spc>g=0,<space|0.5spc>h=0,<space|0.5spc>i=<frac|<sqrt|6>|2>,<space|0.5spc>j=-<frac|1|2>,<space|0.5spc>k=-<frac|1|2>,<space|0.5spc>l=-<frac|1|2><right|]><right|]><leqno>(6)>

      <axiomtype|List List Equation Fraction Polynomial AlgebraicNumber >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      \;
    </input>
  </session>>

  <section|Derivatives of the 2D Basis functions>

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

  <section|Derivatives of the 3D Basis functions>

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
      h:=operator 'h
    </input>

    <\output>
      <with|mode|math|math-display|true|h<leqno>(3)>

      <axiomtype|BasicOperator >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      a:= -2*(1+r)/(s+t) - 1
    </input>

    <\output>
      <with|mode|math|math-display|true|<frac|-t-s-2r-2|t+s><leqno>(4)>

      <axiomtype|Fraction Polynomial Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      b:=2*(1+s)/(1-t) - 1
    </input>

    <\output>
      <with|mode|math|math-display|true|<frac|-t-2s-1|t-1><leqno>(5)>

      <axiomtype|Fraction Polynomial Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      expr:=sqrt(8)*f(a)*g(b)*(1-b)^i*h(c)*(1-c)**(i+j)
    </input>

    <\output>
      <with|mode|math|math-display|true|2<sqrt|2>f<left|(><frac|-t-s-2r-2|t+s><right|)>g<left|(><frac|-t-2s-1|t-1><right|)>h<left|(>c<right|)><left|(>-c+1<right|)><rsup|<left|(>j+i<right|)>><frac|2t+2s|t-1><rsup|i><leqno>(6)>

      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      D(expr,r)
    </input>

    <\output>
      <with|mode|math|math-display|true|-<frac|4<sqrt|2>g<left|(><frac|-t-2s-1|t-1><right|)>h<left|(>c<right|)><left|(>-c+1<right|)><rsup|<left|(>j+i<right|)>><frac|2t+2s|t-1><rsup|i>f<rsub|
      ><rsup|,><left|(><frac|-t-s-2r-2|t+s><right|)>|t+s><leqno>(7)>

      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      dfds:=D(expr,s);
    </input>

    <\output>
      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      dfdsden:=denominator dfds
    </input>

    <\output>
      <with|mode|math|math-display|true|t<rsup|3>+<left|(>2s-1<right|)>t<rsup|2>+<left|(>s<rsup|2>-2s<right|)>t-s<rsup|2><leqno>(20)>

      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      dfds*dfdsden
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|(>-4t<rsup|2>-8s*t-4s<rsup|2><right|)><sqrt|2>f<left|(><frac|-t-s-2r-2|t+s><right|)>h<left|(>c<right|)><left|(>-c+1<right|)><rsup|<left|(>j+i<right|)>><frac|2t+2s|t-1><rsup|i>g<rsub|
      ><rsup|,><left|(><frac|-t-2s-1|t-1><right|)>+<left|(><left|(>4r+4<right|)>t-4r-4<right|)><sqrt|2>g<left|(><frac|-t-2s-1|t-1><right|)>h<left|(>c<right|)><left|(>-c+1<right|)><rsup|<left|(>j+i<right|)>><frac|2t+2s|t-1><rsup|i>f<rsub|
      ><rsup|,><left|(><frac|-t-s-2r-2|t+s><right|)>+<left|(>4i*t<rsup|2>+8i*s*t+4i*s<rsup|2><right|)><sqrt|2>f<left|(><frac|-t-s-2r-2|t+s><right|)>g<left|(><frac|-t-2s-1|t-1><right|)>h<left|(>c<right|)><left|(>-c+1<right|)><rsup|<left|(>j+i<right|)>><frac|2t+2s|t-1><rsup|<left|(>i-1<right|)>><leqno>(18)>

      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      dfdt:=D(expr,t);
    </input>

    <\output>
      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      dfdtden:=denominator dfdt
    </input>

    <\output>
      <with|mode|math|math-display|true|t<rsup|4>+<left|(>2s-2<right|)>t<rsup|3>+<left|(>s<rsup|2>-4s+1<right|)>t<rsup|2>+<left|(>-2s<rsup|2>+2s<right|)>t+s<rsup|2><leqno>(23)>

      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      dfdt*dfdtden
    </input>

    <\output>
      <with|mode|math|math-display|true|<left|(><left|(>4s+4<right|)>t<rsup|2>+<left|(>8s<rsup|2>+8s<right|)>t+4s<rsup|3>+4s<rsup|2><right|)><sqrt|2>f<left|(><frac|-t-s-2r-2|t+s><right|)>h<left|(>c<right|)><left|(>-c+1<right|)><rsup|<left|(>j+i<right|)>><frac|2t+2s|t-1><rsup|i>g<rsub|
      ><rsup|,><left|(><frac|-t-2s-1|t-1><right|)>+<left|(><left|(>4r+4<right|)>t<rsup|2>+<left|(>-8r-8<right|)>t+4r+4<right|)><sqrt|2>g<left|(><frac|-t-2s-1|t-1><right|)>h<left|(>c<right|)><left|(>-c+1<right|)><rsup|<left|(>j+i<right|)>><frac|2t+2s|t-1><rsup|i>f<rsub|
      ><rsup|,><left|(><frac|-t-s-2r-2|t+s><right|)>+<left|(><left|(>-4i*s-4i<right|)>t<rsup|2>+<left|(>-8i*s<rsup|2>-8i*s<right|)>t-4i*s<rsup|3>-4i*s<rsup|2><right|)><sqrt|2>f<left|(><frac|-t-s-2r-2|t+s><right|)>g<left|(><frac|-t-2s-1|t-1><right|)>h<left|(>c<right|)><left|(>-c+1<right|)><rsup|<left|(>j+i<right|)>><frac|2t+2s|t-1><rsup|<left|(>i-1<right|)>><leqno>(24)>

      <axiomtype|Expression Integer >
    </output>

    <\input|<with|color|red|<with|mode|math|\<rightarrow\>> >>
      \;
    </input>
  </session>>

  <section|DG Schemes>

  <subsection|Notation>

  Let <with|mode|math|l<rsub|i>> be the <with|mode|math|i>th Lagrange
  interpolation polynomial and <with|mode|math|\<b-u\>=(u<rsub|i>)> be the
  vector of nodal coefficients of our approximated function
  <with|mode|math|<wide|u|~>(x)=u<rsub|i>l<rsub|i>(x)>. (<with|mode|math|u>
  without subscript denotes a continuous function <with|mode|math|u>.)

  <with|mode|math|F\<subset\>\<partial\>T<rsub|k>> represents the discrete
  faces of the element <with|mode|math|T<rsub|k>>, and
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

  <subsection|Advection Equation>

  <subsubsection|Weak DG>

  We're discretizing <with|mode|math|u<rsub|t>+v\<cdot\>\<nabla\>u=0>. The
  analytic solution is

  <\equation*>
    u(x,t)=u<rsub|0>(x-v*t).
  </equation*>

  Recall

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<nabla\>\<cdot\>(v*u*\<varphi\>)>|<cell|=>|<cell|\<nabla\>\<cdot\>(v*u)\<varphi\>+v*u\<cdot\>\<nabla\>\<varphi\>=(v\<cdot\>\<nabla\>*u)\<varphi\>+v*u\<cdot\>\<nabla\>\<varphi\>.>>>>
  </eqnarray*>

  Using the summation convention, we find

  <\eqnarray*>
    <tformat|<table|<row|<cell|0>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|T<rsub|k>>(v\<cdot\>\<nabla\>u)\<varphi\>>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|D<rsub|k>>v*u\<cdot\>\<nabla\>\<varphi\>+<big|int><rsub|T<rsub|k>>\<nabla\>\<cdot\>(v*u*\<varphi\>)>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|D<rsub|k>>v*u\<cdot\>\<nabla\>\<varphi\>+<big|int><rsub|\<partial\>T<rsub|k>>v*u\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|\<approx\>>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|D<rsub|k>>v*u\<cdot\>\<nabla\>\<varphi\>+<big|int><rsub|\<partial\>T<rsub|k>>(v*u)<rsup|\<ast\>>\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|\<approx\>>|<cell|<big|int><rsub|T<rsub|k>>\<partial\><rsub|t>u<rsub|i>l<rsub|i>l<rsub|j>-<big|int><rsub|T<rsub|k>><matrix|<tformat|<table|<row|<cell|a<rsub|1>u<rsub|i>l<rsub|i>>>|<row|<cell|a<rsub|2>u<rsub|i>l<rsub|i>>>>>>\<cdot\><matrix|<tformat|<table|<row|<cell|\<partial\><rsub|x<rsub|2>>l<rsub|j>>>|<row|<cell|\<partial\><rsub|x<rsub|2>>l<rsub|j>>>>>>+<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><big|int><rsub|F><matrix|<tformat|<table|<row|<cell|(v<rsub|1>u<rsub|i>)<rsup|\<ast\>>l<rsub|i><rsup|F>l<rsub|j><rsup|F>>>|<row|<cell|(v<rsub|2>u<rsub|i>)<rsup|\<ast\>>l<rsub|i><rsup|F>l<rsub|j><rsup|F>>>>>>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|>>|<row|<cell|>|<cell|=>|<cell|>>>>
  </eqnarray*>

  <subsubsection|Strong DG>

  <\eqnarray*>
    <tformat|<table|<row|<cell|0>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|D<rsub|k>>a*u\<cdot\>\<nabla\>\<varphi\>+<big|int><rsub|\<partial\>T<rsub|k>>(v*u)<rsup|\<ast\>>\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|D<rsub|k>>(v\<cdot\>\<nabla\>u)\<varphi\>-<big|int><rsub|\<partial\>T<rsub|k>>(v*u-{v*u})\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|D<rsub|k>>(v\<cdot\>\<nabla\>u)\<varphi\>-<big|int><rsub|\<partial\>T<rsub|k>>v<matrix|<tformat|<table|<row|<cell|<frac|1|2>>>|<row|<cell|-<frac|1|2>>>>>>\<cdot\><matrix|<tformat|<table|<row|<cell|u<rsup|->>>|<row|<cell|u<rsup|+>>>>>>\<varphi\>\<cdot\>n>>>>
  </eqnarray*>

  <subsection|Wave Equation>

  <subsubsection|Weak DG>

  \;

  We have <with|mode|math|u<rsub|t t>-\<Delta\>u=0>, so

  <\eqnarray*>
    <tformat|<table|<row|<cell|u<rsub|t>-\<nabla\>\<cdot\>v>|<cell|=>|<cell|0,>>|<row|<cell|v<rsub|t>-\<nabla\>u>|<cell|=>|<cell|0.>>>>
  </eqnarray*>

  Note that <with|mode|math|v> is a vector. \ Then, using the summation
  convention,

  <\eqnarray*>
    <tformat|<table|<row|<cell|0>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|T<rsub|k>>(\<nabla\>\<cdot\>v)\<varphi\>>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|D<rsub|k>>v\<cdot\>\<nabla\>\<varphi\>-<big|int><rsub|T<rsub|k>>\<nabla\>\<cdot\>(v\<varphi\>)>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|D<rsub|k>>v\<cdot\>\<nabla\>\<varphi\>-<big|int><rsub|\<partial\>T<rsub|k>>v\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|\<approx\>>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|D<rsub|k>>v\<cdot\>\<nabla\>\<varphi\>-<big|int><rsub|\<partial\>T<rsub|k>>v<rsup|\<ast\>>\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|\<approx\>>|<cell|<big|int><rsub|T<rsub|k>>\<partial\><rsub|t>u<rsub|i>l<rsub|i>l<rsub|j>+<big|int><rsub|T<rsub|k>><matrix|<tformat|<table|<row|<cell|[v<rsub|i>]<rsub|1>l<rsub|i>>>|<row|<cell|[v<rsub|i>]<rsub|2>l<rsub|i>>>>>>\<cdot\><matrix|<tformat|<table|<row|<cell|\<partial\><rsub|x<rsub|2>>l<rsub|j>>>|<row|<cell|\<partial\><rsub|x<rsub|2>>l<rsub|j>>>>>>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><big|int><rsub|F><matrix|<tformat|<table|<row|<cell|[v<rsub|i>]<rsup|\<ast\>><rsub|1>l<rsub|i><rsup|F>l<rsub|j><rsup|F>>>|<row|<cell|[v<rsub|i>]<rsup|\<ast\>><rsub|2>l<rsub|i><rsup|F>l<rsub|i><rsup|F>>>>>>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|M<rsub|i
    j><rsup|k>\<partial\><rsub|t>u<rsub|i>+S<rsub|i
    j,\<partial\>x<rsub|1>><rsup|k>[v<rsub|i>]<rsub|1>+S<rsub|i
    j,\<partial\>x<rsub|2>><rsup|k>[v<rsub|i>]<rsub|2>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><left|[>[v<rsub|i>]<rsup|\<ast\>><rsub|1>M<rsub|i
    j><rsup|F>n<rsub|1>+[v<rsub|i>]<rsup|\<ast\>><rsub|2>M<rsub|i
    j><rsup|F>n<rsub|2><right|]>>>|<row|<cell|>|<cell|=>|<cell|(M<rsup|k>)<rsup|T>\<partial\><rsub|t>\<b-u\>+(S<rsub|\<partial\>x<rsub|1>><rsup|k>)<rsup|T>\<b-v\><rsub|1>+(S<rsub|\<partial\>x<rsub|2>><rsup|k>)<rsup|T>\<b-v\><rsub|2>-<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><left|[>[v<rsub|F>]<rsup|\<ast\>><rsub|1>(M<rsup|F>)<rsup|T>n<rsub|1>+[v<rsub|F>]<rsup|\<ast\>><rsub|2>(M<rsup|F>)<rsup|T>n<rsub|2><right|]>>>>>
  </eqnarray*>

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

  <subsubsection|Strong DG>

  First equation:

  <\eqnarray*>
    <tformat|<table|<row|<cell|0>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|D<rsub|k>>v\<cdot\>\<nabla\>\<varphi\>-<big|int><rsub|\<partial\>T<rsub|k>>v<rsup|\<ast\>>\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|D<rsub|k>>(\<nabla\>\<cdot\>v)\<varphi\>+<big|int><rsub|\<partial\>T<rsub|k>>(v-v<rsup|\<ast\>>)\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|D<rsub|k>>(\<nabla\>\<cdot\>v)\<varphi\>+<frac|1|2><big|int><rsub|\<partial\>T<rsub|k>>[v]\<varphi\>\<cdot\>n>>>>
  </eqnarray*>

  Second equation:

  <\eqnarray*>
    <tformat|<table|<row|<cell|0>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>v<rsub|t>\<cdot\>\<psi\>+<big|int><rsub|T<rsub|k>>u\<nabla\>\<cdot\>\<psi\>-<big|int><rsub|\<partial\>T<rsub|k>>u<rsup|\<ast\>>\<psi\>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>v<rsub|t>\<cdot\>\<psi\>-<big|int><rsub|T<rsub|k>>(\<nabla\>u)\<psi\>+<big|int><rsub|\<partial\>T<rsub|k>>(u-u<rsup|\<ast\>>)\<psi\>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>v<rsub|t>\<cdot\>\<psi\>-<big|int><rsub|T<rsub|k>>(\<nabla\>u)\<psi\>+<frac|1|2><big|int><rsub|\<partial\>T<rsub|k>>[u]\<psi\>\<cdot\>n>>>>
  </eqnarray*>

  <subsection|Heat Equation>

  Begin with

  <\eqnarray*>
    <tformat|<table|<row|<cell|u<rsub|t>-\<nabla\>\<cdot\>(<sqrt|a>q)>|<cell|=>|<cell|0,>>|<row|<cell|q-<sqrt|a>\<nabla\>u>|<cell|=>|<cell|0,>>>>
  </eqnarray*>

  so

  <\eqnarray*>
    <tformat|<table|<row|<cell|<big|int><rsub|T>[u<rsub|t>-\<nabla\>\<cdot\>(<sqrt|a>q)]\<varphi\>>|<cell|=>|<cell|0>>|<row|<cell|<big|int><rsub|T>u<rsub|t>\<varphi\>+<big|int><rsub|T><sqrt|a>*q\<cdot\>\<nabla\>\<varphi\>-<big|int><rsub|T>\<nabla\>\<cdot\>(<sqrt|a>*q*\<varphi\>)>|<cell|=>|<cell|0>>|<row|<cell|<big|int><rsub|T>u<rsub|t>\<varphi\>+<big|int><rsub|T><sqrt|a>*q\<cdot\>\<nabla\>\<varphi\>-<big|int><rsub|\<partial\>T><sqrt|a>*q*\<varphi\>\<cdot\>n>|<cell|=>|<cell|0>>|<row|<cell|<big|int><rsub|T>u<rsub|t>\<varphi\>+<big|int><rsub|T><sqrt|a>*q\<cdot\>\<nabla\>\<varphi\>-<big|int><rsub|\<partial\>T>(<sqrt|a>*q)<rsup|\<ast\>>\<varphi\>\<cdot\>n>|<cell|=>|<cell|0,>>>>
  </eqnarray*>

  and

  <\eqnarray*>
    <tformat|<table|<row|<cell|<big|int><rsub|T>[q-<sqrt|a>\<nabla\>u]\<cdot\>\<psi\>>|<cell|=>|<cell|0>>|<row|<cell|<big|int><rsub|T>q\<cdot\>\<psi\>-<big|int><rsub|T><sqrt|a>\<psi\>\<cdot\>(\<nabla\>u)>|<cell|=>|<cell|0>>|<row|<cell|<big|int><rsub|T>q\<cdot\>\<psi\>+<big|int><rsub|T>\<nabla\>\<cdot\>(<sqrt|a>*u)\<psi\>-<big|int><rsub|T>\<nabla\>\<cdot\>(<sqrt|a>*u\<psi\>)>|<cell|=>|<cell|0>>|<row|<cell|<big|int><rsub|T>q\<cdot\>\<psi\>+<big|int><rsub|T>\<nabla\>\<cdot\>(<sqrt|a>*u)\<psi\>-<big|int><rsub|\<partial\>T>n\<cdot\>(<sqrt|a>*u)<rsup|\<ast\>>\<psi\>>|<cell|=>|<cell|0.>>>>
  </eqnarray*>

  <section|Quadrature Rules>

  Golub-Welsch recursion:

  <\equation*>
    p<rsub|n>=(\<alpha\><rsub|n>x+\<beta\><rsub|n>)p<rsub|n-1>-\<gamma\><rsub|n>p<rsub|n-2>
  </equation*>

  Hesthaven-Warburton ``recursion'':

  <\equation*>
    x*p<rsub|n>=a<rsub|n>p<rsub|n-1>+b<rsub|n>p<rsub|n>+a<rsub|n+1>p<rsub|n+1>
  </equation*>

  Solve for <with|mode|math|a<rsub|n>> and <with|mode|math|b<rsub|n>> from
  G-W:

  <\eqnarray*>
    <tformat|<table|<row|<cell|p<rsub|n+1>>|<cell|=>|<cell|\<alpha\><rsub|n+1>x*p<rsub|n>+\<beta\><rsub|n+1>p<rsub|n>-\<gamma\><rsub|n+1>p<rsub|n-1>>>|<row|<cell|\<alpha\><rsub|n+1>x*p<rsub|n>>|<cell|=>|<cell|p<rsub|n+1>-\<beta\><rsub|n+1>p<rsub|n>+\<gamma\><rsub|n+1>p<rsub|n-1>>>|<row|<cell|x*p<rsub|n>>|<cell|=>|<cell|<frac|1|\<alpha\><rsub|n+1>>p<rsub|n+1>-<frac|\<beta\><rsub|n+1>|\<alpha\><rsub|n+1>>p<rsub|n>+<frac|\<gamma\><rsub|n+1>|\<alpha\><rsub|n+1>>p<rsub|n-1>>>|<row|<cell|>|<cell|=>|<cell|<wide*|<frac|\<gamma\><rsub|n+1>|\<alpha\><rsub|n+1>>|\<wide-underbrace\>><rsub|a<rsub|n>>p<rsub|n-1><wide*|-<frac|\<beta\><rsub|n+1>|\<alpha\><rsub|n+1>>|\<wide-underbrace\>><rsub|b<rsub|n>>p<rsub|n>+<wide*|<frac|1|\<alpha\><rsub|n+1>>|\<wide-underbrace\>><rsub|a<rsub|n+1>>p<rsub|n+1>.>>>>
  </eqnarray*>

  <section|Upwind Fluxes>

  <subsection|DIY Upwind Fluxes/Riemann Solvers>

  Start with a linear hyperbolic system:

  <\equation*>
    \<b-u\><rsub|t>+<big|sum><rsub|i>A<rsub|i>\<partial\><rsub|i>\<b-u\>=0
  </equation*>

  <\itemize>
    <item>Pick a unit normal <math|\<b-n\>> of an interface where the states
    <math|\<b-u\><rsup|->> and <math|\<b-u\><rsup|+>> meet.

    <item>Compute

    <\equation*>
      A<rsup|\<pm\>>(\<b-n\>)\<assign\><big|sum><rsub|i>n<rsub|i>A<rsub|i><rsup|\<pm\>>
    </equation*>

    and diagonalize it to <math|D<rsup|\<pm\>>=V*<rsup|\<pm\>>A<rsup|\<pm\>>(\<b-n\>)*(V<rsup|\<pm\>>)<rsup|T>>,
    resulting in eigenvalues

    <\equation*>
      \<lambda\><rsub|1><rsup|\<pm\>>\<leqslant\>\<cdots\>\<leqslant\>\<lambda\><rsub|k><rsup|\<pm\>>\<leqslant\>0\<leqslant\>\<lambda\><rsub|k+1><rsup|\<pm\>>\<leqslant\>\<cdots\>\<leqslant\>\<lambda\><rsub|n><rsup|\<pm\>>.
    </equation*>

    Call <math|\<b-s\><rsup|\<pm\>>\<assign\>V<rsup|\<pm\>>\<b-u\><rsup|\<pm\>>>.

    <item>Consider the space-time diagram of Figure <reference|fig:fluxfan>.
    Each eigenvalue is imagined to be the speed of a <em|shock> emanating
    from the point under consideration, and each shock separating one
    <em|state> from another. The left- and rightmost state are
    <math|\<b-s\><rsup|->> and <math|\<b-s\><rsup|+>>, of course. The next
    state from the left is labeled <math|\<b-s\><rsup|\<ast\>>>, the one
    after that <math|\<b-s\><rsup|\<ast\>\<ast\>>>, up to
    <math|\<b-s\><rsup|\<ast\>(n-1)>>. These states are a-priori unknown. In
    addition, the fluxes <math|(D\<b-s\>)<rsup|\<ast\>>,\<ldots\>,(D\<b-s\>)<rsup|\<ast\>(n-1)>>
    are also unknown: For the moment, there are <math|2(n-1)> unknowns.

    <item>The Rankine-Hugoniot condition

    <\equation*>
      \<lambda\><rsub|i><rsup|\<pm\>>=<frac|(D\<b-s\>)<rsup|*\<ast\>(i)>-(D\<b-s\>)<rsup|*\<ast\>(i-1)>|\<b-s\><rsup|*\<ast\>(i)>-\<b-s\><rsup|*\<ast\>(i-1)>>
    </equation*>

    has to hold across each such interface. For convenience, set
    <math|\<b-s\><rsup|\<ast\>(0)>=\<b-s\><rsup|->> and
    <math|\<b-s\><rsup|\<ast\>(n)>=\<b-s\><rsup|+>>. This will provide
    <math|n> equations.

    Naturally, eigenvalues larger than zero (``right-travelling'') use their
    <math|\<cdot\><rsup|->> values, whereas left-travelling values smaller
    than zero use their <math|\<cdot\><rsup|+>> values.

    <item>We label <math|\<lambda\><rsub|k>> that eigenvalue whose state
    straddles zero the line of zero speed--that includes the case of
    <math|\<lambda\><rsub|k>=0>. If there are several zero eigenvalues (can
    that even happen?), pick <math|k> as large as possible. It is assumed
    that the number <math|k> is the same on both sides of the interface, and
    therefore it carries no <math|\<pm\>> superscript.

    <item>The numerical flux, i.e. the solution that we seek, is the flux
    <math|(D\<b-s\>)<rsup|\<ast\>(k)>> that straddles zero.

    <item>For non-zero-straddling states, the unknowns
    <math|(D\<b-s\>)<rsup|\<ast\>(i)>> are found as follows:

    <\itemize>
      <item>If <math|i\<less\>k>, <math|(D\<b-s\>)<rsup|\<ast\>(i)>=D<rsup|->\<b-s\><rsup|\<ast\>(i)>>.

      <item>If <math|i\<gtr\>k>, <math|(D\<b-s\>)<rsup|\<ast\>(i)>=D<rsup|+>\<b-s\><rsup|\<ast\>(i)>>.
    </itemize>

    This takes <math|n-2> unknowns out of the picture, leaving us with
    <math|2(n-1)-(n-2)=n> unknowns.

    <item>The case of a zero eigenvalue deserves special mention, even though
    its handling is technically no different than before: The states
    <math|\<b-s\><rsup|\<ast\>(k-1)>> and <math|\<b-s\><rsup|\<ast\>(k)>>
    fall cleanly to the left and right of zero, so we can use

    <\eqnarray*>
      <tformat|<table|<row|<cell|(D\<b-s\>)<rsup|\<ast\>(k-1)>>|<cell|=>|<cell|D<rsup|->\<b-s\><rsup|\<ast\>(k-1)>,>>|<row|<cell|(D\<b-s\>)<rsup|\<ast\>(k)>>|<cell|=>|<cell|D<rsup|+>\<b-s\><rsup|\<ast\>(k)>.>>>>
    </eqnarray*>

    Observe that we end up with only <math|n-1> unknowns (namely all the
    <math|{\<b-s\><rsup|\<ast\>(i)>}<rsub|i=1><rsup|n-1>>), but we still have
    <math|n> equations. It seems, however, that the resulting system of
    equations has rank <math|n-1>, and so a unique solution can be
    determined. (<with|color|red|Why is that, rigorously?>)

    Lastly, observe that the <math|k>th Rankine-Hugoniot condition dictates

    <\equation*>
      (D\<b-s\>)<rsup|\<ast\>(k-1)>=(D\<b-s\>)<rsup|\<ast\>(k)>,
    </equation*>

    ensuring consistency across the zero-speed shock.
  </itemize>

  I believe that the above can also be applied in the case of degenerate
  eigenvalues, but haven't checked in detail.

  <big-figure|<postscript|fluxfan.fig|9cm|||||>|<label|fig:fluxfan>Space-Time
  Diagram of a flux fan.>

  <subsection|Comments by Akil>

  <subsubsection|Zero eigenvalues>

  I'm not entirely sure that the case of zero eigenvalues actually comes out
  well. Let's consider the following: adopting your notation above, let's
  assume we have <math|n> eigenvalues for each element,
  <math|\<lambda\><rsub|i><rsup|\<pm\>>>, <math|i = 1,2,\<ldots\>n>. Let
  <math|N>, <math|M>, and <math|P> represent the number of negative, zero,
  and positive eigenvalues, respectively. (Recall that we're assuming that
  <math|sign(\<lambda\><rsub|i><rsup|+>) = sign(\<lambda\><rsub|i><rsup|->)>
  for all <math|i>.) We have <math|N+M+P=n>, and we assume at least two of
  these numbers are nonzero. (If only one of them is nonzero, upwinding is
  trivial.)

  \;

  We start with the <math|2*(n-1)> unknowns
  <math|<left|{><with|math-font-series|bold|s><rsup|\<asterisk\>(i)>,
  <left|(>D*<with|math-font-series|bold|s><right|)><rsup|\<asterisk\>(i)><right|}><rsub|i=1><rsup|n-1>>.\ 

  \;

  I'm adopting the convention that positive eigenvalues correspond to
  right-traveling waves. We have the following Rankine-Hugoniot conditions:

  <\equation>
    <tabular|<tformat|<cwith|3|3|3|3|cell-halign|l>|<table|<row|<cell|\<lambda\><rsup|-><rsub|i>>|<cell|=>|<cell|<frac|(D\<b-s\>)<rsup|*\<ast\>(i)>-(D\<b-s\>)<rsup|*\<ast\>(i-1)>|\<b-s\><rsup|*\<ast\>(i)>-\<b-s\><rsup|*\<ast\>(i-1)>>,>|<cell|>|<cell|i
    = 1, 2, \<ldots\>,N>>|<row|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>>|<row|<cell|0>|<cell|=>|<cell|(D\<b-s\>)<rsup|*\<ast\>(i)>-(D\<b-s\>)<rsup|*\<ast\>(i-1)>,>|<cell|>|<cell|i
    = N+1, \<ldots\>,N+Z>>|<row|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>>|<row|<cell|\<lambda\><rsup|+><rsub|i>>|<cell|=>|<cell|<frac|(D\<b-s\>)<rsup|*\<ast\>(i)>-(D\<b-s\>)<rsup|*\<ast\>(i-1)>|\<b-s\><rsup|*\<ast\>(i)>-\<b-s\><rsup|*\<ast\>(i-1)>>,>|<cell|>|<cell|i
    = N+Z+1, \<ldots\>,n>>>>><right|}> \ n \ (linear)
    equations<label|eq:rh-all>
  </equation>

  \;

  \;

  If I understand what we're doing, we consider the unknowns <math|<left|{>
  <left|(>D*<with|math-font-series|bold|s><right|)><rsup|\<asterisk\>(i)><right|}><rsub|i=1,
  2, \<ldots\><with|math-font-series|bold|N-1>,
  <with|math-font-series|bold|N+Z+1>, \<ldots\>, n>> as anisotropic
  (directionally-biased) unknowns with explicit connections to the
  <math|<with|math-font-series|bold|s><rsup|\<asterisk\>(i)>> given by the
  auxilliary equations

  <\equation>
    <tabular|<tformat|<table|<row|<cell|(D\<b-s\>)<rsup|*\<ast\>(i)> =
    D<rsup|->\<b-s\><rsup|*\<ast\>(i)>>|<cell|>|<cell|i = 1, 2,
    \<ldots\>,N-1>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|(D\<b-s\>)<rsup|*\<ast\>(i)>
    = D<rsup|+>\<b-s\><rsup|*\<ast\>(i)>>|<cell|>|<cell|i = N+Z+1, \<ldots\>,
    n>>>>><right|}> \ N+P-2 \ \ (linear) equations<label|eq:rh-aux>
  </equation>

  \;

  However, I don't see what we do for the unknowns which make contact with
  the zero eigenvalue(s). You mention that if <math|Z=1> with
  <math|\<lambda\><rsub|k>=0> we would have\ 

  <\equation*>
    <tabular|<tformat|<table|<row|<cell|(D\<b-s\>)<rsup|\<ast\>(k)>>|<cell|<above|=|<with|mode|text|NO!>>>|<cell|D<rsup|+>\<b-s\><rsup|\<ast\>(k)>.>>>>>
  </equation*>

  But then I'd also claim that\ 

  <\equation*>
    <tabular|<tformat|<table|<row|<cell|(D\<b-s\>)<rsup|\<ast\>(k-1)>>|<cell|<above|=|<with|mode|text|NO!>>>|<cell|D<rsup|->\<b-s\><rsup|\<ast\>(k-1)>.>>>>>
  </equation*>

  I cannot think of a reason why we'd treat
  <with|mode|math|(D\<b-s\>)<rsup|\<ast\>(k-1)>> any differently from
  <with|mode|math|(D\<b-s\>)<rsup|\<ast\>(k)>> since I don't think it's the
  case that <with|mode|math|(D\<b-s\>)<rsup|\<ast\>(k)>> `belongs' to
  <math|\<lambda\><rsub|k>=0> in any way. If <math|Z=0>, then indeed
  (<reference|eq:rh-aux>) gives us <math|N+P-2 = n-2> equations to add to the
  <math|n> equations in (<reference|eq:rh-all>) giving <math|2*n-2> equations
  to couple the <math|2(n-1)> unknowns. But if <math|Z=1>, then
  (<reference|eq:rh-aux>) gives us <math|N+P-2=n-3> equations to couple with
  the <math|n> from (<reference|eq:rh-all>). That's <math|2*n-3> equations
  for <math|2*n-2> unknowns. Conversely, if instead we adopted the convention

  <\equation*>
    <tabular|<tformat|<table|<row|<cell|(D\<b-s\>)<rsup|\<ast\>(k)>>|<cell|<above|=|<with|mode|text|YES!>>>|<cell|D<rsup|+>\<b-s\><rsup|\<ast\>(k)>>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|(D\<b-s\>)<rsup|\<ast\>(k-1)>>|<cell|<above|=|<with|mode|text|YES!>>>|<cell|D<rsup|+>\<b-s\><rsup|\<ast\>(k)>,>>>>>
  </equation*>

  then we'd have 2 equations plus <math|N+P-2=n-3> from
  (<reference|eq:rh-aux>) plus <math|n> from (<reference|eq:rh-all>) giving
  us <math|2*n-1> equations for <math|2*n-2> unknowns.\ 

  \;

  In general, if we don't adopt any directional bias for states adjacent to
  the zero eigenvalue, then for <math|Z\<gtr\>0> zero eigenvalues, we'd have
  <math|2*n-2-Z> equations for <math|2*n-2> unknowns, and if we do adopt
  directional bias for states adjacent to the zero eigenvalue, then we'd have
  <math|2n-Z> equations for <math|2n-2> unknowns (If
  <math|\<lambda\><rsub|k>=\<lambda\><rsub|k+1>=0>, I don't see how you can
  adopt a directional bias for <with|mode|math|(D\<b-s\>)<rsup|\<ast\>(k)>>.)

  \;

  You make the comment above that ``We simply regard
  <math|(D\<b-s\>)<rsup|\<ast\>(k)>> as another unkown of the system, which,
  in addition to <math|{\<b-s\><rsup|*\<ast\>(i)>}<rsub|1><rsup|n-1>>, brings
  us to <math|n> unknowns.''

  \;

  If I understand this correctly, this indeed does make <math|n> equations
  and <math|n> unknowns, but you treat <with|mode|math|(D\<b-s\>)<rsup|\<ast\>(k)>>
  differently than <with|mode|math|(D\<b-s\>)<rsup|\<ast\>(k+1)>> (non
  directionally-biased vs. directionally biased) and, again, I don't see how
  you can consistently make this choice given that, to me,
  <with|mode|math|(D\<b-s\>)<rsup|\<ast\>(k+1)>> is no more directionally
  unbiased than <with|mode|math|(D\<b-s\>)<rsup|\<ast\>(k)>>.\ 

  <subsubsection|Degenerate eigenvalues>

  \;

  Another comment: degenerate eigenvalues should be ok: let's suppose that
  <math|\<lambda\><rsub|k>=\<lambda\><rsub|k+1>>. Then there is an
  intermediate state <math|<with|math-font-series|bold|s><rsup|\<asterisk\>(k)>>
  separating <math|<with|math-font-series|bold|s><rsup|\<asterisk\>(k-1)>>
  and <math|<with|math-font-series|bold|s><rsup|\<asterisk\>(k+1)>>. This
  will give us\ 

  <\equation*>
    <tabular|<tformat|<table|<row|<cell|\<lambda\><rsub|k>=<frac|(D\<b-s\>)<rsup|*\<ast\>(k+1)>-(D\<b-s\>)<rsup|*\<ast\>(k)>|\<b-s\><rsup|*\<ast\>(k+1)>-\<b-s\><rsup|*\<ast\>(k)>>>>|<row|<cell|>>|<row|<cell|\<lambda\><rsub|k>=<frac|(D\<b-s\>)<rsup|*\<ast\>(k)>-(D\<b-s\>)<rsup|*\<ast\>(k-1)>|\<b-s\><rsup|*\<ast\>(k)>-\<b-s\><rsup|*\<ast\>(k-1)>>>>>>>
  </equation*>

  Using these equations to eliminate the unknowns
  <with|mode|math|(D\<b-s\>)<rsup|*\<ast\>(k)>> and
  <with|mode|math|\<b-s\><rsup|*\<ast\>(k)>> gives us the Rankine-Hugoniot
  condition relating <with|mode|math|\<b-s\><rsup|*\<ast\>(k-1)>> to
  <with|mode|math|\<b-s\><rsup|*\<ast\>(k+1)>>:

  <\equation*>
    \<lambda\><rsub|k>=<frac|(D\<b-s\>)<rsup|*\<ast\>(k+1)>-(D\<b-s\>)<rsup|*\<ast\>(k-1)>|\<b-s\><rsup|*\<ast\>(k+1)>-\<b-s\><rsup|*\<ast\>(k-1)>>.
  </equation*>

  This is the same condition one would get if we pretended that states
  <with|mode|math|(D\<b-s\>)<rsup|*\<ast\>(k)>> and
  <with|mode|math|\<b-s\><rsup|*\<ast\>(k)>>, the states straddled by the
  shocks of the same speed, did not exist and merrily went about our
  upwinding.\ 

  \;

  Note that this also means that a <math|Z>-fold degeneracy in the zero
  eigenvalue isn't a problem either, as long as we can sort out the simple
  zero eigenvalue case.

  <subsection|More random comments by Akil>

  We seem to be pretty confident that if there does not exist any 0
  eigenvalue, then everything is fine. Kittens meow, rainbows shine, and the
  world is happy. But we've also established that we don't know what's going
  on when <math|\<exists\> \ \ \<lambda\>=0>. However, let's boil things down
  to the simplest case we can. We know degenerate eigenvalues pose no problem
  (see my comment last time), and multiple simple (+) or (<math|->)
  eigenvalues are likewise fine. So let's consider this model case:

  <\equation*>
    A<rsup|\<pm\>>(<wide|<with|math-font-series|bold|n>|^>):<with|math-font|Bbb*|R><rsup|3>\<rightarrow\><with|math-font|Bbb*|R><rsup|3>,
    \ \ \<lambda\><rsup|\<pm\>><rsub|1>\<less\>0,
    \<lambda\><rsup|\<pm\>><rsub|2>=0, \ \ \ and
    \ \ \ \<lambda\><rsup|\<pm\>><rsub|3>\<gtr\>0.
  </equation*>

  We assume the greatest generality possible here:
  <math|\<lambda\><rsub|\<cdot\>><rsup|\<pm\>>> can be different magnitudes,
  but they have the same sign. In line with a strong hyperbolic system, we
  assume that <math|A<rsup|\<pm\>>> is diagonalizable as

  <\equation*>
    A<rsup|\<pm\>>=V<rsup|\<pm\>>\<Lambda\><rsup|\<pm\>><left|(>V<rsup|\<pm\>><right|)><rsup|T>,
  </equation*>

  where <math|\<Lambda\><rsup|\<pm\>>> are diagonal matrices containing the
  <math|\<lambda\><rsup|\<pm\>><rsub|i>>. We introduce the eigenvectors

  <\equation*>
    (V<rsup|\<pm\>>) =<left|(><tabular|<tformat|<table|<row|<cell|<with|math-font-series|bold|v><rsub|1><rsup|\<pm\>>>|<cell|<with|math-font-series|bold|v><rsub|2><rsup|\<pm\>>>|<cell|<with|math-font-series|bold|v><rsub|3><rsup|\<pm\>>>>>>><right|)>
  </equation*>

  Here, all eigenvalues are simple, and <math|<left|{><with|math-font-series|bold|v><rsub|i><rsup|\<pm\>><right|}><rsub|i=1><rsup|3>>
  span <math|<with|math-font|Bbb*|R><rsup|3>> for each (+) or (<math|->).
  <with|font-series|bold|Note:> there is no reason to assume that, e.g.
  <math|<left|{><with|math-font-series|bold|v><rsub|1><rsup|+>,
  <with|math-font-series|bold|v><rsub|2><rsup|->,
  <with|math-font-series|bold|v><rsub|3><rsup|-><right|}>> span
  <math|<with|math-font|Bbb*|R><rsup|3>>. Or we shall not assume so here, at
  least. Although, to be fair, if this is not the case, then we can have the
  case of a genuine shock forming, which is kind of ridiculous. Of course, we
  have that <math|\<lambda\><rsub|i><rsup|\<pm\>>> is associated with
  <math|<with|math-font-series|bold|v><rsub|i><rsup|\<pm\>>>. We also assume
  that\ 

  <\equation*>
    <tabular|<tformat|<table|<row|<cell|<with|math-font-series|bold|u><rsup|->=<big|sum><rsub|i=1><rsup|3>\<mu\><rsup|-><rsub|i><with|math-font-series|bold|v><rsub|i><rsup|->>|<cell|>|<cell|<with|math-font-series|bold|u><rsup|+>=<big|sum><rsub|i=1><rsup|3>\<mu\><rsub|i><rsup|+><with|math-font-series|bold|v><rsub|i><rsup|+>>>>>>,
  </equation*>

  which we can do since the <math|<with|math-font-series|bold|v><rsub|i>>'s
  span. And of course, I just realized that the
  <math|\<mu\><rsub|i><rsup|\<pm\>>> are the entries in
  <math|<with|math-font-series|bold|s><rsup|\<pm\>><rsub|i>>. Go math.\ 

  \;

  We introduce the intermediate states <math|<with|math-font-series|bold|u><rsup|\<asterisk\>>>
  and <math|<with|math-font-series|bold|u><rsup|\<asterisk\>\<asterisk\>>>,
  with their appropriate fluxes <math|(A<with|math-font-series|bold|u>)<rsup|\<asterisk\>>>
  and <math|(A<with|math-font-series|bold|u>)<rsup|\<asterisk\>\<asterisk\>>>.
  All four of these, at the moment, are unknown.\ 

  \;

  We note that at the end of the day, we want some flux\ 

  <\equation*>
    (A<with|math-font-series|bold|u>)<rsup|#>
  </equation*>

  (yes that's horrible notation; leave me alone! wahhh!)

  \;

  All of our algorithms will attempt to find this quantity. Now, we consider
  three possible algorithms for upwind flux computations.\ 

  <subsubsection|Stoopid upwinding>

  The motivation here is simple: the 255 view of upwinding is the following:
  the evolution equation for <math|u<rsup|->> should not allow boundary
  conditions that impose <math|<with|math-font-series|bold|v><rsub|1><rsup|->>;
  similarly, the evolution equation for <math|u<rsup|+>> should not allow
  boundary conditions that impose <math|<with|math-font-series|bold|v><rsub|3><rsup|+>>.
  This is algorithmically consistent with vanilla upwinding
  <math|u<rsub|t>=a*u<rsub|x>>. The algorithm then is the following:

  <\enumerate>
    <item>We simply set <math|<with|math-font-series|bold|u><rsup|\<asterisk\>>=<with|math-font-series|bold|u><rsup|->-\<mu\><rsup|-><rsub|2>*<with|math-font-series|bold|v><rsup|-><rsub|2>-\<mu\><rsup|-><rsub|1><with|math-font-series|bold|v><rsup|-><rsub|1>=\<mu\><rsup|-><rsub|3><with|math-font-series|bold|v><rsup|-><rsub|3>>.
    Also set <math|<with|math-font-series|bold|u><rsup|\<asterisk\>\<asterisk\>>=<with|math-font-series|bold|u><rsup|+>-\<mu\><rsup|+><rsub|2>*<with|math-font-series|bold|v><rsup|+><rsub|2>-\<mu\><rsup|+><rsub|3><with|math-font-series|bold|v><rsup|+><rsub|3>=\<mu\><rsub|1><rsup|+><with|math-font-series|bold|v><rsup|+><rsub|1>>.\ 

    <item>Set <math|(A<with|math-font-series|bold|u>)<rsup|\<asterisk\>>=A<rsup|-><with|math-font-series|bold|u><rsup|\<asterisk\>>>
    and <math|(A*<with|math-font-series|bold|u>)<rsup|\<asterisk\>\<asterisk\>>=A<rsup|+>*<with|math-font-series|bold|u><rsup|\<asterisk\>\<asterisk\>>>.\ 

    <item>Use the flux <math|(A<with|math-font-series|bold|s>)<rsup|#>=(A<with|math-font-series|bold|u>)<rsup|\<asterisk\>>+(A*<with|math-font-series|bold|u>)<rsup|\<asterisk\>\<asterisk\>>=\<mu\><rsup|-><rsub|3><with|math-font-series|bold|v><rsup|-><rsub|3>+\<mu\><rsub|1><rsup|+><with|math-font-series|bold|v><rsup|+><rsub|1>>.
  </enumerate>

  \;

  Advantages of this flux:

  <\itemize>
    <item>It's easy

    <item>It's very intuitive

    <item>It's consistent with usual upwinding if <math|A<rsup|+>=A<rsup|->>.
  </itemize>

  Disadvantages:

  <\itemize>
    <item>There is no motivation for it other than, ``it's upwinding''.

    <item>Where'r the R-H conditions? Actually, these are
    <with|font-series|bold|almost> exactly those conditions: see below....
  </itemize>

  <subsubsection|R-H Conditions with Directional Bias>

  In this formulation, we impose the Rankine-Hugoniot conditions, which in
  this case read

  <\equation*>
    <tabular|<tformat|<cwith|3|3|1|1|cell-halign|r>|<table|<row|<cell|\<lambda\><rsub|1><rsup|->(<with|math-font-series|bold|u><rsup|\<asterisk\>>-<with|math-font-series|bold|u><rsup|->)>|<cell|=>|<cell|<left|(>A<with|math-font-series|bold|u><right|)><rsup|\<asterisk\>>-A<rsup|-><with|math-font-series|bold|u><rsup|->>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|(A<with|math-font-series|bold|u><right|)><rsup|\<asterisk\>>>|<cell|=>|<cell|(A<with|math-font-series|bold|u><right|)><rsup|\<asterisk\>\<asterisk\>>>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|\<lambda\><rsub|3><rsup|+>(<with|math-font-series|bold|u><rsup|\<asterisk\>\<asterisk\>>-<with|math-font-series|bold|u><rsup|+>)>|<cell|=>|<cell|<left|(>A<with|math-font-series|bold|u><right|)><rsup|\<asterisk\>\<asterisk\>>-A<rsup|+><with|math-font-series|bold|u><rsup|+>>>>>>
  </equation*>

  \;

  However, we also impose `directional bias', which means that intermediate
  states which are adjacent to, but not overlapping, the zero-speed shock
  inherit the operators from their respective sides. In math, this reads

  <\equation*>
    <tabular|<tformat|<table|<row|<cell|(A<with|math-font-series|bold|u><right|)><rsup|\<asterisk\>>=A<rsup|-><with|math-font-series|bold|u><rsup|\<asterisk\>>>|<cell|>|<cell|(A<with|math-font-series|bold|u><right|)><rsup|\<asterisk\>\<asterisk\>>=A<rsup|+><with|math-font-series|bold|u><rsup|\<asterisk\>\<asterisk\>>.>>>>>
  </equation*>

  This is very interesting set of conditions: it seemingly over-determines
  the system, and the R-H conditions become\ 

  <\equation*>
    <tabular|<tformat|<cwith|3|3|1|1|cell-halign|r>|<table|<row|<cell|<left|(>A<rsup|->-\<lambda\><rsub|1><rsup|->I<right|)><left|(><with|math-font-series|bold|u><rsup|\<asterisk\>>-<with|math-font-series|bold|u><rsup|-><right|)>>|<cell|=>|<cell|0>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|A<rsup|-><with|math-font-series|bold|u><rsup|\<asterisk\>>>|<cell|=>|<cell|A<rsup|+><with|math-font-series|bold|u><rsup|\<asterisk\>\<asterisk\>>>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|<left|(>A<rsup|+>-\<lambda\><rsub|3><rsup|+>I<right|)><left|(><with|math-font-series|bold|u><rsup|\<asterisk\>\<asterisk\>>-<with|math-font-series|bold|u><rsup|+><right|)>>|<cell|=>|<cell|0>>>>>
  </equation*>

  \;

  This means that the quantity <with|mode|math|<with|math-font-series|bold|u><rsup|\<asterisk\>>-<with|math-font-series|bold|u><rsup|->>
  *must* be a multiple of <math|<with|math-font-series|bold|v><rsup|-><rsub|1>>.
  I.e.,\ 

  <\equation*>
    <with|math-font-series|bold|u><rsup|\<asterisk\>>=\<nu\><rsub|1><with|math-font-series|bold|v><rsub|1><rsup|->+\<mu\><rsub|2><rsup|-><with|math-font-series|bold|v><rsub|2><rsup|->+\<mu\><rsub|3><rsup|-><with|math-font-series|bold|v><rsub|3><rsup|->,
  </equation*>

  for some unknown scalar <math|\<nu\><rsub|1>>. A similar argument leads us
  to\ 

  <\equation*>
    <with|math-font-series|bold|u><rsup|\<asterisk\>\<asterisk\>>=\<mu\><rsup|+><rsub|1><with|math-font-series|bold|v><rsub|1><rsup|+>+\<mu\><rsub|2><rsup|+><with|math-font-series|bold|v><rsub|2><rsup|+>+\<nu\><rsub|3><with|math-font-series|bold|v><rsub|3><rsup|+>.
  </equation*>

  It is worth stressing that our unknown vectors
  <math|<with|math-font-series|bold|u><rsup|\<asterisk\>>> and
  <math|<with|math-font-series|bold|u><rsup|\<asterisk\>\<asterisk\>>>,
  previously a full 6 unknowns, are now merely two unknowns,
  <math|\<nu\><rsub|1>> and <math|\<nu\><rsub|3>>. This will be true for any
  size of the matrices <math|A<rsup|\<pm\>>>.\ 

  \;

  <with|font-shape|italic|Note that this has a strikingly similar flavor to
  `stoopid upwinding': in that case, we would have <math|\<nu\><rsub|1>=0>
  and <math|\<nu\><rsub|3>=0>. If we impose directional bias, `Stoopid
  upwinding' automatically satisfies all but one of the R-H conditions: the
  one at <math|\<lambda\>=0>.>

  \;

  Of course, we are still overdetermined: we have one R-H condition left:
  three equations, two unknowns. What's left? The <math|\<lambda\>=0>
  conditions boils down to\ 

  <\equation*>
    \<nu\><rsub|1>\<lambda\><rsub|1><rsup|->*<with|math-font-series|bold|v><rsub|1><rsup|->-\<nu\><rsub|3>*\<lambda\><rsub|3><rsup|+><with|math-font-series|bold|v><rsub|3><rsup|+>=\<mu\><rsub|1><rsup|+>\<lambda\><rsub|1><rsup|+><with|math-font-series|bold|v><rsub|1><rsup|+>-\<mu\><rsub|3><rsup|->\<lambda\><rsub|3><rsup|-><with|math-font-series|bold|v><rsub|3><rsup|->,
  </equation*>

  where the unknowns are <math|\<nu\><rsub|1>> and <math|\<nu\><rsub|3>>.
  Note that we *need* this equation to be satisifed independent of the
  unknowns <math|\<mu\><rsub|1><rsup|+>> and <math|\<mu\><rsub|3><rsup|->>,
  which are dependent on the incoming waves
  <math|<with|math-font-series|bold|u><rsup|\<pm\>>>, over which we have no
  control. This means that we require that
  <math|span<left|{><with|math-font-series|bold|v><rsub|1><rsup|+>,
  <with|math-font-series|bold|v><rsub|3><rsup|-><right|}>\<subset\>span<left|{><with|math-font-series|bold|v><rsub|1><rsup|->,<with|math-font-series|bold|v><rsub|3><rsup|+><right|}>>.
  Note that since <math|<with|math-font-series|bold|v><rsub|1><rsup|+>\<perp\><with|math-font-series|bold|v><rsub|3><rsup|+>>,
  and <math|<with|math-font-series|bold|v><rsub|1><rsup|->\<perp\><with|math-font-series|bold|v><rsub|3><rsup|->>,
  then we must have that <math|<with|math-font-series|bold|v><rsub|1><rsup|+>\<parallel\><with|math-font-series|bold|v><rsub|1><rsup|+>>
  and <math|<with|math-font-series|bold|v><rsub|3><rsup|->\<parallel\><with|math-font-series|bold|v><rsub|3><rsup|->>.
  Then (and only then!) will we be able to solve for the fluxes we wish for
  any values of <math|\<mu\><rsup|\<pm\>><rsub|\<cdot\>>>.\ 

  \;

  In this simple case, our condition states that
  <math|<with|math-font-series|bold|v><rsub|1><rsup|+>=c<rsub|1>*<with|math-font-series|bold|v><rsub|1><rsup|->>
  and <math|<with|math-font-series|bold|v><rsub|3><rsup|+>=c<rsub|3>*<with|math-font-series|bold|v><rsub|3><rsup|->>.
  I.e., that left traveling waves stay left-traveling waves and
  right-traveling waves do the same. Then we have that\ 

  <\equation*>
    <tabular|<tformat|<table|<row|<cell|\<nu\><rsub|1>=c<rsub|1>*\<mu\><rsub|1><rsup|+>*<frac|\<lambda\><rsub|1><rsup|+>|\<lambda\><rsub|1><rsup|->>>|<cell|>|<cell|\<nu\><rsub|3>=\<mu\><rsub|3><rsup|-><frac|\<lambda\><rsub|3><rsup|->|c<rsub|3>*\<lambda\><rsub|3><rsup|+>>>>>>>,
  </equation*>

  which uniquely determines the flux as\ 

  <\equation*>
    <left|(>A<with|math-font-series|bold|u><right|)><rsup|#>=A<rsup|-><with|math-font-series|bold|u><rsup|\<asterisk\>>=A<rsup|+><with|math-font-series|bold|u><rsup|\<asterisk\>\<asterisk\>>=\<nu\><rsub|1>\<lambda\><rsub|1><rsup|-><with|math-font-series|bold|v><rsub|1><rsup|->+\<mu\><rsub|3><rsup|->\<lambda\><rsub|3><rsup|-><with|math-font-series|bold|v><rsub|3><rsup|->=\<mu\><rsup|+><rsub|1>\<lambda\><rsub|1><rsup|+><with|math-font-series|bold|v><rsub|1><rsup|+>+\<nu\><rsub|3>\<lambda\><rsub|3><rsup|+><with|math-font-series|bold|v><rsub|3><rsup|+>.
  </equation*>

  \;

  The above considerations lead us to the following theorem:

  <\theorem>
    (Well-posedness of Rankine-Hugoniot directionally-biased upwinding)

    Let <math|A<rsup|\<pm\>>(<wide|<with|math-font-series|bold|n>|^>)>
    represent the (discontinuous) operator of a multidimensional strongly
    hyperbolic linear wave equation at an element interface. Assume
    diagonalizations of the form\ 

    <\equation*>
      <tabular|<tformat|<table|<row|<cell|A<rsup|\<pm\>>=V<rsup|\<pm\>>\<Lambda\><rsup|\<pm\>><left|(>V<rsup|\<pm\>><right|)><rsup|T>,>>|<row|<cell|>>|<row|<cell|V<rsup|\<pm\>>=<left|(><tabular|<tformat|<table|<row|<cell|<with|math-font-series|bold|v><rsup|\<pm\>><rsub|1>>|<cell|<with|math-font-series|bold|v><rsub|2><rsup|\<pm\>>>|<cell|\<cdots\>>|<cell|<with|math-font-series|bold|v><rsub|n><rsup|\<pm\>>>>>>><right|)>>>>>>
    </equation*>

    \;

    Let the evolving <math|<with|math-font-series|bold|u>\<in\><with|math-font|Bbb*|R><rsup|n>>.
    The known values of <math|<with|math-font-series|bold|u>> at the element
    interfaces admit the decomposition\ 

    <\equation*>
      <tabular|<tformat|<table|<row|<cell|<with|math-font-series|bold|u><rsup|->=<big|sum><rsub|i=1><rsup|n>\<mu\><rsub|i><rsup|-><with|math-font-series|bold|v><rsub|i><rsup|->>|<cell|>|<cell|<with|math-font-series|bold|u><rsup|+>=<big|sum><rsub|i=1><rsup|n>\<mu\><rsub|i><rsup|+><with|math-font-series|bold|v><rsub|i><rsup|+>.>>>>>
    </equation*>

    Assume that each diagonal matrix <math|\<Lambda\><rsup|\<pm\>>> contains
    eigenvalues <math|\<lambda\><rsub|i><rsup|\<pm\>>> such that the
    following quantities are well-defined:

    <\equation*>
      <tabular|<tformat|<table|<row|<cell|<with|math-font|cal|P><rsup|+>=<left|{>i:
      \<lambda\><rsup|+><rsub|i>\<gtr\>0<right|}>>|<cell|>|<cell|<with|math-font|cal|N><rsup|+>=<left|{>i:
      \<lambda\><rsup|+><rsub|i>\<less\>0<right|}>>|<cell|>|<cell|<with|math-font|cal|Z><rsup|+>=<left|{>i:
      \<lambda\><rsup|+><rsub|i>=0<right|}>>>|<row|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>>|<row|<cell|<with|math-font|cal|P><rsup|->=<left|{>i:
      \<lambda\><rsup|-><rsub|i>\<gtr\>0<right|}>>|<cell|>|<cell|<with|math-font|cal|N><rsup|->=<left|{>i:
      \<lambda\><rsup|-><rsub|i>\<less\>0<right|}>>|<cell|>|<cell|<with|math-font|cal|Z><rsup|->=<left|{>i:
      \<lambda\><rsup|-><rsub|i>=0<right|}>>>|<row|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>>|<row|<cell|P
      =<left|\|><with|math-font|cal|P><rsup|+><right|\|>=<left|\|><with|math-font|cal|P><rsup|-><right|\|>>|<cell|>|<cell|N
      =<left|\|><with|math-font|cal|N><rsup|+><right|\|>=<left|\|><with|math-font|cal|N><rsup|-><right|\|>>|<cell|>|<cell|Z
      =<left|\|><with|math-font|cal|Z><rsup|+><right|\|>=<left|\|><with|math-font|cal|Z><rsup|-><right|\|>.>>>>>
    </equation*>

    Note that this gives us <math|P+Z+N=n>. Associate with each eigenvalue
    <math|\<lambda\><rsup|\<pm\>><rsub|i>> the eigenvector
    <math|<with|math-font-series|bold|v><rsub|i><rsup|\<pm\>>>. Define the
    following vector subspaces of <math|<with|math-font|Bbb*|R><rsup|n>>:

    <\equation*>
      <tabular|<tformat|<table|<row|<cell|<with|math-font|cal|W><rsub|<with|math-font|cal|P>><rsup|+>=span<left|{><with|math-font-series|bold|v><rsub|i><rsup|+>:i\<in\><with|math-font|cal|P><rsup|+><right|}>>|<cell|>|<cell|<with|math-font|cal|W><rsub|<with|math-font|cal|N>><rsup|+>=span<left|{><with|math-font-series|bold|v><rsub|i><rsup|+>:i\<in\><with|math-font|cal|N><rsup|+><right|}>>|<cell|>|<cell|<with|math-font|cal|W><rsub|<with|math-font|cal|Z>><rsup|+>=span<left|{><with|math-font-series|bold|v><rsub|i><rsup|+>:i\<in\><with|math-font|cal|Z><rsup|+><right|}>>>|<row|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>>|<row|<cell|<with|math-font|cal|W><rsub|<with|math-font|cal|P>><rsup|->=span<left|{><with|math-font-series|bold|v><rsub|i><rsup|->:i\<in\><with|math-font|cal|P><rsup|-><right|}>>|<cell|>|<cell|<with|math-font|cal|W><rsub|<with|math-font|cal|N>><rsup|->=span<left|{><with|math-font-series|bold|v><rsub|i><rsup|->:i\<in\><with|math-font|cal|N><rsup|-><right|}>>|<cell|>|<cell|<with|math-font|cal|W><rsub|<with|math-font|cal|Z>><rsup|->=span<left|{><with|math-font-series|bold|v><rsub|i><rsup|->:i\<in\><with|math-font|cal|Z><rsup|-><right|}>>>>>>
    </equation*>

    \;

    Define the <math|(n-1)> intermediate states
    <math|<left|{><with|math-font-series|bold|u><rsup|\<asterisk\>(i)><right|}><rsub|i=1><rsup|n-1>>
    and the associated fluxes <math|<left|{><left|(>A<with|math-font-series|bold|u><right|)><rsup|\<asterisk\>(i)><right|}><rsup|n-1><rsub|i=1>>
    such that <math|<with|math-font-series|bold|u><rsup|\<asterisk\>(i)>> and
    <with|mode|math|<left|(>A<with|math-font-series|bold|u><right|)><rsup|\<asterisk\>(i)>>
    `lies between' eigenvalues <math|\<lambda\><rsub|i><rsup|\<pm\>>> and
    <math|\<lambda\><rsub|i+1><rsup|\<pm\>>> (see figure
    <reference|fig:fluxfan>). Now define the unsuperscripted eigenvalues as\ 

    <\equation*>
      <tabular|<tformat|<table|<row|<cell|<left|{>\<lambda\><rsub|i><right|}><rsub|i=1><rsup|N>=sort<left|{>\<lambda\><rsub|i><rsup|->:
      i\<in\><with|math-font|cal|N><rsup|-><right|}>>>|<row|<cell|>>|<row|<cell|<left|{>\<lambda\><rsub|i><right|}><rsub|i=N+1><rsup|N+Z>=sort<left|{>\<lambda\><rsub|i><rsup|->:
      i\<in\><with|math-font|cal|Z><rsup|-><right|}>>>|<row|<cell|>>|<row|<cell|<left|{>\<lambda\><rsub|i><right|}><rsub|i=N+1><rsup|N+Z>=sort<left|{>\<lambda\><rsub|i><rsup|+>:
      i\<in\><with|math-font|cal|P><rsup|+><right|}>>>>>>
    </equation*>

    \;

    Define the numerical flux as\ 

    <\equation*>
      <left|(>A<with|math-font-series|bold|u><right|)><rsup|#>\<assign\><left|(>A<with|math-font-series|bold|u><right|)><rsup|\<asterisk\>(q)>,
    </equation*>

    where <math|q = argmin<left|{><left|\|>\<lambda\><rsub|i><right|\|><right|}><rsub|i=1><rsup|N>>.\ 

    \;

    Assume the following conditions:

    <\enumerate>
      <item><math|Z\<gtr\>0>

      <item>Imposition of the Rankine-Hugoniot condition across eigenvalues:

      <\equation*>
        \<lambda\><rsub|i><left|(><with|math-font-series|bold|u><rsup|\<asterisk\>(i)>-<with|math-font-series|bold|u><rsup|\<asterisk\>(i-1)><right|)>
        = <left|(>A<with|math-font-series|bold|u><right|)><rsup|\<asterisk\>(i)>-<left|(>A<with|math-font-series|bold|u><right|)><rsup|\<asterisk\>(i-1)><hspace|1cm>
        \<forall\> \ \ i = 1, 2, \<ldots\>n,
      </equation*>

      where we define <math|<with|math-font-series|bold|u><rsup|\<asterisk\>(0)>=<with|math-font-series|bold|u><rsup|->>
      and <math|<with|math-font-series|bold|u><rsup|\<asterisk\>(n)>=<with|math-font-series|bold|u><rsup|+>>.\ 

      <item>Directional biasing:

      <\equation*>
        (A<with|math-font-series|bold|u>)<rsup|\<asterisk\>(i)>=<left|{><tabular|<tformat|<table|<row|<cell|A<rsup|-><with|math-font-series|bold|u><rsup|\<asterisk\>(i)>>|<cell|>|<cell|if>|<cell|>|<cell|i
        = 1, 2, \<ldots\>,N>>|<row|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>>|<row|<cell|A<rsup|+><with|math-font-series|bold|u><rsup|\<asterisk\>(i)>>|<cell|>|<cell|if>|<cell|>|<cell|i
        = n-1, n-2, \<ldots\>,n-P>>>>>
      </equation*>

      <item><math|<with|math-font|cal|W><rsup|+><rsub|<with|math-font|cal|P>>=<with|math-font|cal|W><rsup|-><rsub|<with|math-font|cal|P>>>
      and <math|<with|math-font|cal|W><rsub|<with|math-font|cal|N>><rsup|->=<with|math-font|cal|W><rsup|+><rsub|<with|math-font|cal|N>>>
      \ (<math|\<Longrightarrow\><with|math-font|cal|W<rsub|Z><rsup|->>=<with|math-font|cal|W><rsub|<with|math-font|cal|Z>><rsup|+>>)
    </enumerate>

    \;

    Then the numerical flux <math|<left|(>A<with|math-font-series|bold|u><right|)><rsup|#>>
    exists and is unique. It is given by\ 

    <\equation*>
      <tabular|<tformat|<table|<row|<cell|<left|(>A<with|math-font-series|bold|u><right|)><rsup|#>>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N>\<nu\><rsub|i>\<lambda\><rsub|i><with|math-font-series|bold|v><rsub|i><rsup|->+<big|sum><rsub|i=n-P+1><rsup|n>\<mu\><rsub|i><rsup|->\<lambda\><rsub|i><with|math-font-series|bold|v><rsub|i><rsup|->>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N>\<mu\><rsup|+><rsub|i>\<lambda\><rsub|i><with|math-font-series|bold|v><rsub|i><rsup|+>+<big|sum><rsub|i=n-P+1><rsup|n>\<nu\><rsub|i>\<lambda\><rsub|i><with|math-font-series|bold|v><rsub|i><rsup|+>,>>>>>
    </equation*>

    \;

    where the <math|N> unknowns <math|<left|{>\<nu\><rsub|i><right|}><rsub|i=1><rsup|N>>
    satisfy the rank <math|N> system\ 

    <\equation*>
      <left|(><tabular|<tformat|<table|<row|<cell|\<lambda\><rsub|1><with|math-font-series|bold|v><rsub|1><rsup|->>|<cell|\<lambda\><rsub|2><with|math-font-series|bold|v><rsub|2><rsup|->>|<cell|>|<cell|\<cdots\>>|<cell|>|<cell|\<lambda\><rsub|N><with|math-font-series|bold|v><rsub|N><rsup|->>>>>><right|)><left|(><tabular|<tformat|<table|<row|<cell|\<nu\><rsub|1>>>|<row|<cell|\<nu\><rsub|2>>>|<row|<cell|\<vdots\>>>|<row|<cell|\<nu\><rsub|N>>>>>><right|)>=<big|sum><rsub|i=1><rsup|N>\<mu\><rsup|+><rsub|i>\<lambda\><rsub|i><with|math-font-series|bold|v><rsub|i><rsup|+>,
    </equation*>

    and the <math|P> unknowns <math|<left|{>\<nu\><rsub|i><right|}><rsub|i=n-P+1><rsup|n>>
    satisfy the rank <math|P> system\ 

    <\equation*>
      <left|(><tabular|<tformat|<table|<row|<cell|\<lambda\><rsub|n-P+1><with|math-font-series|bold|v><rsub|n-P+1><rsup|+>>|<cell|\<lambda\><rsub|n-P+2><with|math-font-series|bold|v><rsub|n-P+2><rsup|+>>|<cell|>|<cell|\<cdots\>>|<cell|>|<cell|\<lambda\><rsub|n><with|math-font-series|bold|v><rsub|n><rsup|+>>>>>><right|)><left|(><tabular|<tformat|<table|<row|<cell|\<nu\><rsub|n-P+1>>>|<row|<cell|\<nu\><rsub|n-P+2>>>|<row|<cell|\<vdots\>>>|<row|<cell|\<nu\><rsub|n>>>>>><right|)>=<big|sum><rsub|i=n-P+1><rsup|n>\<mu\><rsup|-><rsub|i>\<lambda\><rsub|i><with|math-font-series|bold|v><rsub|i><rsup|->.
    </equation*>
  </theorem>

  Note that this theorem even takes into account degeneracies in any
  eigenvalue.

  <subsection|Deriving an Upwind Flux for the Wave Equation>

  We're dealing with

  <\equation*>
    u<rsub|t t>=c<rsup|2>(x)\<Delta\>u,
  </equation*>

  which we rewrite into conservation form:

  <\eqnarray*>
    <tformat|<table|<row|<cell|u<rsub|t>-c(x)\<nabla\>\<cdot\>\<b-v\>>|<cell|=>|<cell|0>>|<row|<cell|\<b-v\><rsub|t>-c(x)\<nabla\>u>|<cell|=>|<cell|0>>>>
  </eqnarray*>

  and from there into matrix form:

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<b-w\><rsub|t>+c(x)<left|[><wide*|<matrix|<tformat|<table|<row|<cell|0>|<cell|-1>|<cell|0>>|<row|<cell|-1>|<cell|>|<cell|>>|<row|<cell|0>|<cell|>|<cell|>>>>>|\<wide-underbrace\>><rsub|A<rsub|x>\<assign\>>\<b-w\><rsub|x>+<wide*|<matrix|<tformat|<table|<row|<cell|0>|<cell|0>|<cell|-1>>|<row|<cell|0>|<cell|>|<cell|>>|<row|<cell|-1>|<cell|>|<cell|>>>>>|\<wide-underbrace\>><rsub|A<rsub|y>\<assign\>>\<b-w\><rsub|y><right|]>>|<cell|=>|<cell|\<b-0\>>>>>
  </eqnarray*>

  where we set <math|\<b-w\>\<assign\>(u,\<b-v\>)>.\ 

  (see <with|font-family|tt|doc/maxima/wave.mac> and
  <with|font-family|tt|doc/maxima/myhelpers.mac> for Maxima scripts for code
  to find the upwind flux.)

  We find:

  <\eqnarray*>
    <tformat|<table|<row|<cell|(\<Pi\>*u)<rsup|\<ast\>>>|<cell|=>|<cell|\<b-n\>\<cdot\><average|c\<b-v\>>+<frac|[c*u]<rsup|->-[c*u]<rsup|+>|2>>>|<row|<cell|(\<Pi\>\<b-v\>)<rsup|\<ast\>>>|<cell|=>|<cell|\<b-n\><average|c*u>+<frac|1|2>(\<b-n\>\<otimes\>\<b-n\>)\<cdot\>[c<rsup|->\<b-v\><rsup|->-c<rsup|+>\<b-v\><rsup|+>].>>>>
  </eqnarray*>

  <subsection|Deriving an Upwind Flux for the 2D Advection Equation>

  We're dealing with

  <\equation*>
    u<rsub|t>=\<b-v\>\<cdot\>\<nabla\>u,
  </equation*>

  which we rewrite into matrix conservation form as

  <\equation*>
    u<rsub|t>=v<rsub|1>u<rsub|x>+v<rsub|2>u<rsub|y>.
  </equation*>

  We find <math|A(\<b-n\>)=\<b-n\>\<cdot\>\<b-v\>>, and therefore

  <\equation*>
    (A(\<b-n\>)u)<rsup|\<ast\>>=<choice|<tformat|<table|<row|<cell|\<b-n\>\<cdot\>\<b-v\>u<rsup|->>|<cell|<with|mode|text|if
    <math|\<b-n\>\<cdot\>\<b-v\>\<geqslant\>0>
    (outflow)>,>>|<row|<cell|\<b-n\>\<cdot\>\<b-v\>u<rsup|+>>|<cell|<with|mode|text|if
    <math|\<b-n\>\<cdot\>\<b-v\>\<less\>0> (inflow)>.>>>>>
  </equation*>

  <section|Tim's funky curvilinear element stuff>

  We're discretizing <with|mode|math|u<rsub|t>+v\<cdot\>\<nabla\>u=0>. Recall

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<nabla\>\<cdot\>(v*u*\<varphi\>)>|<cell|=>|<cell|\<nabla\>\<cdot\>(v*u)\<varphi\>+v*u\<cdot\>\<nabla\>\<varphi\>=(v\<cdot\>\<nabla\>*u)\<varphi\>+v*u\<cdot\>\<nabla\>\<varphi\>.>>>>
  </eqnarray*>

  Using the summation convention, we find

  <\eqnarray*>
    <tformat|<table|<row|<cell|0>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|T<rsub|k>>(v\<cdot\>\<nabla\>u)\<varphi\>>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|D<rsub|k>>v*u\<cdot\>\<nabla\>\<varphi\>+<big|int><rsub|T<rsub|k>>\<nabla\>\<cdot\>(v*u*\<varphi\>)>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|D<rsub|k>>v*u\<cdot\>\<nabla\>\<varphi\>+<big|int><rsub|\<partial\>T<rsub|k>>v*u\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|\<approx\>>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|D<rsub|k>>v*u\<cdot\>\<nabla\>\<varphi\>+<big|int><rsub|\<partial\>T<rsub|k>>(v*u)<rsup|\<ast\>>\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|\<approx\>>|<cell|<big|int><rsub|T<rsub|k>>\<partial\><rsub|t>u<rsub|i>l<rsub|i>l<rsub|j>-<big|int><rsub|T<rsub|k>><matrix|<tformat|<table|<row|<cell|a<rsub|1>u<rsub|i>l<rsub|i>>>|<row|<cell|a<rsub|2>u<rsub|i>l<rsub|i>>>>>>\<cdot\><matrix|<tformat|<table|<row|<cell|\<partial\><rsub|x<rsub|2>>l<rsub|j>>>|<row|<cell|\<partial\><rsub|x<rsub|2>>l<rsub|j>>>>>>+<big|sum><rsub|F\<subset\>\<partial\>T<rsub|k>><big|int><rsub|F><matrix|<tformat|<table|<row|<cell|(v<rsub|1>u<rsub|i>)<rsup|\<ast\>>l<rsub|i><rsup|F>l<rsub|j><rsup|F>>>|<row|<cell|(v<rsub|2>u<rsub|i>)<rsup|\<ast\>>l<rsub|i><rsup|F>l<rsub|j><rsup|F>>>>>>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|>>|<row|<cell|>|<cell|=>|<cell|>>>>
  </eqnarray*>

  <\eqnarray*>
    <tformat|<table|<row|<cell|0>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>-<big|int><rsub|D<rsub|k>>a*u\<cdot\>\<nabla\>\<varphi\>+<big|int><rsub|\<partial\>T<rsub|k>>(v*u)<rsup|\<ast\>>\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|D<rsub|k>>(v\<cdot\>\<nabla\>u)\<varphi\>-<big|int><rsub|\<partial\>T<rsub|k>>(v*u-{v*u})\<varphi\>\<cdot\>n>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|T<rsub|k>>u<rsub|t>\<varphi\>+<big|int><rsub|D<rsub|k>>(v\<cdot\>\<nabla\>u)\<varphi\>-<big|int><rsub|\<partial\>T<rsub|k>>v<matrix|<tformat|<table|<row|<cell|<frac|1|2>>>|<row|<cell|-<frac|1|2>>>>>>\<cdot\><matrix|<tformat|<table|<row|<cell|u<rsup|->>>|<row|<cell|u<rsup|+>>>>>>\<varphi\>\<cdot\>n>>>>
  </eqnarray*>
</body>

<\initial>
  <\collection>
    <associate|info-flag|detailed>
    <associate|page-orientation|portrait>
    <associate|page-type|letter>
    <associate|par-first|0>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|5.3|7>>
    <associate|auto-11|<tuple|5.3.1|7>>
    <associate|auto-12|<tuple|5.3.2|8>>
    <associate|auto-13|<tuple|5.4|9>>
    <associate|auto-14|<tuple|6|9>>
    <associate|auto-15|<tuple|7|10>>
    <associate|auto-16|<tuple|7.1|10>>
    <associate|auto-17|<tuple|1|11>>
    <associate|auto-18|<tuple|7.2|11>>
    <associate|auto-19|<tuple|7.2.1|14>>
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-20|<tuple|7.2.2|14>>
    <associate|auto-21|<tuple|7.3|18>>
    <associate|auto-22|<tuple|7.3.1|?>>
    <associate|auto-23|<tuple|7.3.2|?>>
    <associate|auto-24|<tuple|7.4|?>>
    <associate|auto-25|<tuple|7.5|?>>
    <associate|auto-26|<tuple|8|?>>
    <associate|auto-27|<tuple|8.0.1|?>>
    <associate|auto-28|<tuple|8|?>>
    <associate|auto-29|<tuple|8|?>>
    <associate|auto-3|<tuple|3|3>>
    <associate|auto-4|<tuple|4|4>>
    <associate|auto-5|<tuple|5|6>>
    <associate|auto-6|<tuple|5.1|6>>
    <associate|auto-7|<tuple|5.2|7>>
    <associate|auto-8|<tuple|5.2.1|7>>
    <associate|auto-9|<tuple|5.2.2|7>>
    <associate|auto.1-1|<tuple|1|?|#1>>
    <associate|auto.2-1|<tuple|2|?|#2>>
    <associate|auto.3-1|<tuple|3|?|#3>>
    <associate|auto.4-1|<tuple|4|?|#4>>
    <associate|auto.5-1|<tuple|5|?|#5>>
    <associate|auto.5-2|<tuple|5.1|?|#5>>
    <associate|auto.5-3|<tuple|5.2|?|#5>>
    <associate|auto.5-4|<tuple|5.2.1|?|#5>>
    <associate|auto.5-5|<tuple|5.2.2|?|#5>>
    <associate|auto.5-6|<tuple|5.3|?|#5>>
    <associate|auto.5-7|<tuple|5.3.1|?|#5>>
    <associate|auto.5-8|<tuple|5.3.2|?|#5>>
    <associate|auto.5-9|<tuple|5.4|?|#5>>
    <associate|auto.6-1|<tuple|6|?|#6>>
    <associate|auto.7-1|<tuple|7|?|#7>>
    <associate|auto.7-10|<tuple|7.4|?|#7>>
    <associate|auto.7-11|<tuple|7.5|?|#7>>
    <associate|auto.7-12|<tuple|7.6|?|#7>>
    <associate|auto.7-13|<tuple|7.7|?|#7>>
    <associate|auto.7-2|<tuple|7.1|?|#7>>
    <associate|auto.7-3|<tuple|1|?|#7>>
    <associate|auto.7-4|<tuple|7.2|?|#7>>
    <associate|auto.7-5|<tuple|7.2.1|?|#7>>
    <associate|auto.7-6|<tuple|7.2.2|?|#7>>
    <associate|auto.7-7|<tuple|7.3|?|#7>>
    <associate|auto.7-8|<tuple|7.3.1|?|#7>>
    <associate|auto.7-9|<tuple|7.3.2|?|#7>>
    <associate|auto.8-1|<tuple|8|?|#8>>
    <associate|auto.9-1|<tuple|9|?|#9>>
    <associate|eq:rh-all|<tuple|1|?>>
    <associate|eq:rh-aux|<tuple|2|?>>
    <associate|fig:fluxfan|<tuple|1|11>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|figure>
      <tuple|normal|<label|fig:fluxfan>Space-Time Diagram of a flux
      fan.|<pageref|auto-17>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Mapping
      from 2D equilateral to Unit Coordinates>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Mapping
      from 3D equilateral to Unit Coordinates>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Derivatives
      of the 2D Basis functions> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Derivatives
      of the 3D Basis functions> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>DG
      Schemes> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.5fn>

      <with|par-left|<quote|1.5fn>|5.1<space|2spc>Notation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1.5fn>|5.2<space|2spc>Advection Equation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <with|par-left|<quote|3fn>|5.2.1<space|2spc>Weak DG
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8>>

      <with|par-left|<quote|3fn>|5.2.2<space|2spc>Strong DG
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9>>

      <with|par-left|<quote|1.5fn>|5.3<space|2spc>Wave Equation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10>>

      <with|par-left|<quote|3fn>|5.3.1<space|2spc>Weak DG
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11>>

      <with|par-left|<quote|3fn>|5.3.2<space|2spc>Strong DG
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-12>>

      <with|par-left|<quote|1.5fn>|5.4<space|2spc>Heat Equation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-13>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|6<space|2spc>Quadrature
      Rules> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-14><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|7<space|2spc>Upwind
      Fluxes> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-15><vspace|0.5fn>

      <with|par-left|<quote|1.5fn>|7.1<space|2spc>DIY Upwind Fluxes/Riemann
      Solvers <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-16>>

      <with|par-left|<quote|1.5fn>|7.2<space|2spc>Comments by Akil
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-18>>

      <with|par-left|<quote|3fn>|7.2.1<space|2spc>Zero eigenvalues
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-19>>

      <with|par-left|<quote|3fn>|7.2.2<space|2spc>Degenerate eigenvalues
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-20>>

      <with|par-left|<quote|1.5fn>|7.3<space|2spc>More random comments by
      Akil <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-21>>

      <with|par-left|<quote|3fn>|7.3.1<space|2spc>Stoopid upwinding
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-22>>

      <with|par-left|<quote|3fn>|7.3.2<space|2spc>R-H Conditions with
      Directional Bias <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-23>>

      <with|par-left|<quote|1.5fn>|7.4<space|2spc>Deriving an Upwind Flux for
      the Wave Equation <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-24>>

      <with|par-left|<quote|1.5fn>|7.5<space|2spc>Deriving an Upwind Flux for
      the 2D Advection Equation <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-25>>
    </associate>
  </collection>
</auxiliary>