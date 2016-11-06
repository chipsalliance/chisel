circuit NodeType :
  module NodeType :
    input clock : Clock

    cmem rf : UInt<64>[31]
    node rf_wdata = mux(UInt(0), UInt(0), UInt(0))
    infer mport port = rf[UInt(0)], clock
    port <- rf_wdata
