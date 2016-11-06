circuit Queue :
  module Queue :
    input clock : Clock
    cmem ram : UInt<1>[2]
    infer mport T_107 = ram[UInt(0)], clock
