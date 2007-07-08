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
      \;
    </input>
  </session>>
</body>

<\initial>
  <\collection>
    <associate|page-type|letter>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Cylindrical
      TM Maxwell Cavity Mode> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>