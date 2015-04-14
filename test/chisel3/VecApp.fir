circuit VecApp : 
  module VecApp : 
    input a : UInt(4)
    input i : UInt(8)[4]
    output d : UInt(8)
    
    accessor T_13 = i[a]
    d := T_13