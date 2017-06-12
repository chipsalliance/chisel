circuit HasLoops :
  module HasLoops :
    input i : UInt<1>
    output o : UInt<1>
    wire a : UInt<1>
    wire b : UInt<1>
    a <= and(b,i)
    b <= not(a)
    o <= add(a, UInt(1))
