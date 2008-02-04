<TeXmacs|1.0.6>

<style|generic>

<\body>
  <\with|mode|math>
    <assign|ip|<macro|left-arg|right-arg|subi|supi|<left|(><arg|left-arg>,<arg|right-arg><right|)><rsub|<arg|subi>><rsup|<arg|supi>>>>

    <assign|seminorm|<macro|arg|subi|supi|<left|\|><arg|arg><right|\|><rsub|<arg|subi>><rsup|<arg|supi>>>>

    <assign|norm|<macro|arg|subi|supi|<left|\|\|><arg|arg><right|\|\|><rsub|<arg|subi>><rsup|<arg|supi>>>>

    <assign|nnorm|<macro|arg|subi|supi|<left|\|><space|-0.1fn||><left|\|\|><arg|arg><right|\|\|><space|-0.1fn||><right|\|><rsub|<arg|subi>><rsup|<arg|supi>>>>

    <assign|jump|<macro|arg|<left|llbracket><arg|arg><right|rrbracket>>>

    <assign|average|<macro|arg|<left|{><arg|arg><right|}>>>

    <assign|mean|<macro|index|<superpose|<big|int>| -><rsub|<arg|index>>>>

    <assign|laplace|\<triangle\>>
  </with>

  <assign|eqref|<macro|refname|(<reference|<arg|refname>>)>>
</body>

<\initial>
  <\collection>
    <associate|language|german>
    <associate|preamble|true>
  </collection>
</initial>