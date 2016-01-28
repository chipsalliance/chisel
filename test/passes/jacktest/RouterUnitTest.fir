; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Done!

circuit RouterUnitTester : 
  module Router : 
    input clk : Clock
    input reset : UInt<1>
    output io : {read_routing_table_request : {ready : UInt<1>, flip valid : UInt<1>, flip bits : {addr : UInt<32>}}, read_routing_table_response : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}, load_routing_table_request : {ready : UInt<1>, flip valid : UInt<1>, flip bits : {addr : UInt<32>, data : UInt<32>}}, in : {ready : UInt<1>, flip valid : UInt<1>, flip bits : {header : UInt<8>, body : UInt<64>}}, outs : {flip ready : UInt<1>, valid : UInt<1>, bits : {header : UInt<8>, body : UInt<64>}}[4]}
    
    io.outs[0].bits.body <= UInt<1>("h00")
    io.outs[0].bits.header <= UInt<1>("h00")
    io.outs[0].valid <= UInt<1>("h00")
    io.outs[1].bits.body <= UInt<1>("h00")
    io.outs[1].bits.header <= UInt<1>("h00")
    io.outs[1].valid <= UInt<1>("h00")
    io.outs[2].bits.body <= UInt<1>("h00")
    io.outs[2].bits.header <= UInt<1>("h00")
    io.outs[2].valid <= UInt<1>("h00")
    io.outs[3].bits.body <= UInt<1>("h00")
    io.outs[3].bits.header <= UInt<1>("h00")
    io.outs[3].valid <= UInt<1>("h00")
    io.in.ready <= UInt<1>("h00")
    io.load_routing_table_request.ready <= UInt<1>("h00")
    io.read_routing_table_response.bits <= UInt<1>("h00")
    io.read_routing_table_response.valid <= UInt<1>("h00")
    io.read_routing_table_request.ready <= UInt<1>("h00")
    cmem tbl : UInt<3>[15]
    when reset :
      infer mport T_115 = tbl[UInt<1>("h00")], clk
      T_115 <= UInt<32>("h00")
      infer mport T_118 = tbl[UInt<1>("h01")], clk
      T_118 <= UInt<32>("h00")
      infer mport T_121 = tbl[UInt<2>("h02")], clk
      T_121 <= UInt<32>("h00")
      infer mport T_124 = tbl[UInt<2>("h03")], clk
      T_124 <= UInt<32>("h00")
      infer mport T_127 = tbl[UInt<3>("h04")], clk
      T_127 <= UInt<32>("h00")
      infer mport T_130 = tbl[UInt<3>("h05")], clk
      T_130 <= UInt<32>("h00")
      infer mport T_133 = tbl[UInt<3>("h06")], clk
      T_133 <= UInt<32>("h00")
      infer mport T_136 = tbl[UInt<3>("h07")], clk
      T_136 <= UInt<32>("h00")
      infer mport T_139 = tbl[UInt<4>("h08")], clk
      T_139 <= UInt<32>("h00")
      infer mport T_142 = tbl[UInt<4>("h09")], clk
      T_142 <= UInt<32>("h00")
      infer mport T_145 = tbl[UInt<4>("h0a")], clk
      T_145 <= UInt<32>("h00")
      infer mport T_148 = tbl[UInt<4>("h0b")], clk
      T_148 <= UInt<32>("h00")
      infer mport T_151 = tbl[UInt<4>("h0c")], clk
      T_151 <= UInt<32>("h00")
      infer mport T_154 = tbl[UInt<4>("h0d")], clk
      T_154 <= UInt<32>("h00")
      infer mport T_157 = tbl[UInt<4>("h0e")], clk
      T_157 <= UInt<32>("h00")
      skip
    io.read_routing_table_request.ready <= UInt<1>("h01")
    io.load_routing_table_request.ready <= UInt<1>("h01")
    io.read_routing_table_response.valid <= UInt<1>("h00")
    io.read_routing_table_response.bits <= UInt<1>("h00")
    io.in.ready <= UInt<1>("h01")
    io.outs[0].valid <= UInt<1>("h00")
    io.outs[0].bits.body <= UInt<1>("h00")
    io.outs[0].bits.header <= UInt<1>("h00")
    io.outs[1].valid <= UInt<1>("h00")
    io.outs[1].bits.body <= UInt<1>("h00")
    io.outs[1].bits.header <= UInt<1>("h00")
    io.outs[2].valid <= UInt<1>("h00")
    io.outs[2].bits.body <= UInt<1>("h00")
    io.outs[2].bits.header <= UInt<1>("h00")
    io.outs[3].valid <= UInt<1>("h00")
    io.outs[3].bits.body <= UInt<1>("h00")
    io.outs[3].bits.header <= UInt<1>("h00")
    node T_176 = and(io.read_routing_table_request.valid, io.read_routing_table_response.ready)
    when T_176 :
      io.read_routing_table_request.ready <= UInt<1>("h01")
      infer mport T_178 = tbl[io.read_routing_table_request.bits.addr], clk
      io.read_routing_table_response.valid <= UInt<1>("h01")
      io.read_routing_table_response.bits <= T_178
      skip
    node T_181 = eq(T_176, UInt<1>("h00"))
    node T_182 = and(T_181, io.load_routing_table_request.valid)
    when T_182 :
      io.load_routing_table_request.ready <= UInt<1>("h01")
      infer mport T_184 = tbl[io.load_routing_table_request.bits.addr], clk
      T_184 <= io.load_routing_table_request.bits.data
      node T_186 = eq(reset, UInt<1>("h00"))
      when T_186 :
        printf(clk, UInt<1>(1), "setting tbl(%d) to %d", io.load_routing_table_request.bits.addr, io.load_routing_table_request.bits.data)
        skip
      skip
    node T_188 = eq(T_176, UInt<1>("h00"))
    node T_190 = eq(io.load_routing_table_request.valid, UInt<1>("h00"))
    node T_191 = and(T_188, T_190)
    node T_192 = and(T_191, io.in.valid)
    when T_192 :
      node T_193 = bits(io.in.bits.header, 4, 0)
      infer mport T_194 = tbl[T_193], clk
      when io.outs[T_194].ready :
        io.in.ready <= UInt<1>("h01")
        io.outs[T_194].valid <= UInt<1>("h01")
        io.outs[T_194].bits <- io.in.bits
        infer mport T_215 = tbl[io.in.bits.header], clk
        node T_217 = eq(reset, UInt<1>("h00"))
        when T_217 :
          printf(clk, UInt<1>(1), "got packet to route header %d, data %d, being routed to out(%d) ", io.in.bits.header, io.in.bits.body, T_215)
          skip
        skip
      skip
    
  module RouterUnitTester : 
    input clk : Clock
    input reset : UInt<1>
    output io : {}
    
    inst device_under_test of Router
    device_under_test.io.outs[0].ready <= UInt<1>("h00")
    device_under_test.io.outs[1].ready <= UInt<1>("h00")
    device_under_test.io.outs[2].ready <= UInt<1>("h00")
    device_under_test.io.outs[3].ready <= UInt<1>("h00")
    device_under_test.io.in.bits.body <= UInt<1>("h00")
    device_under_test.io.in.bits.header <= UInt<1>("h00")
    device_under_test.io.in.valid <= UInt<1>("h00")
    device_under_test.io.load_routing_table_request.bits.data <= UInt<1>("h00")
    device_under_test.io.load_routing_table_request.bits.addr <= UInt<1>("h00")
    device_under_test.io.load_routing_table_request.valid <= UInt<1>("h00")
    device_under_test.io.read_routing_table_response.ready <= UInt<1>("h00")
    device_under_test.io.read_routing_table_request.bits.addr <= UInt<1>("h00")
    device_under_test.io.read_routing_table_request.valid <= UInt<1>("h00")
    device_under_test.clk <= clk
    device_under_test.reset <= reset
    reg T_19 : UInt<8>, clk with :
       reset => (reset, UInt<8>("h00"))
    reg T_21 : UInt<1>, clk with :
       reset => (reset, UInt<1>("h00"))
    reg T_23 : UInt<8>, clk with :
       reset => (reset, UInt<8>("h00"))
    reg T_25 : UInt<1>, clk with :
       reset => (reset, UInt<1>("h00"))
    node T_26 = and(T_21, T_25)
    when T_26 :
      node T_28 = eq(reset, UInt<1>("h00"))
      when T_28 :
        printf(clk, UInt<1>(1), "All input and output events completed
")
        skip
      node T_30 = eq(reset, UInt<1>("h00"))
      when T_30 :
        stop(clk, UInt<1>(1), 0)
        skip
      skip
    reg T_32 : UInt<10>, clk with :
       reset => (reset, UInt<10>("h00"))
    node T_34 = tail(add(T_32, UInt<1>("h01")),1)
    T_32 <= T_34
    node T_36 = gt(T_32, UInt<10>("h03e8"))
    when T_36 :
      node T_39 = eq(reset, UInt<1>("h00"))
      when T_39 :
        printf(clk, UInt<1>(1), "Exceeded maximum allowed %d ticks in OrderedDecoupledHWIOTester, If you think code is correct use:
DecoupleTester.max_tick_count = <some-higher-value>
in the OrderedDecoupledHWIOTester subclass
", UInt<10>("h03e8"))
        skip
      node T_41 = eq(reset, UInt<1>("h00"))
      when T_41 :
        stop(clk, UInt<1>(1), 0)
        skip
      skip
    wire T_97 : UInt<1>[54]
    T_97[0] <= UInt<1>("h00")
    T_97[1] <= UInt<1>("h00")
    T_97[2] <= UInt<1>("h00")
    T_97[3] <= UInt<1>("h00")
    T_97[4] <= UInt<1>("h00")
    T_97[5] <= UInt<1>("h00")
    T_97[6] <= UInt<1>("h00")
    T_97[7] <= UInt<1>("h00")
    T_97[8] <= UInt<1>("h00")
    T_97[9] <= UInt<1>("h00")
    T_97[10] <= UInt<1>("h00")
    T_97[11] <= UInt<1>("h00")
    T_97[12] <= UInt<1>("h00")
    T_97[13] <= UInt<1>("h01")
    T_97[14] <= UInt<1>("h01")
    T_97[15] <= UInt<1>("h01")
    T_97[16] <= UInt<1>("h01")
    T_97[17] <= UInt<1>("h00")
    T_97[18] <= UInt<1>("h00")
    T_97[19] <= UInt<1>("h00")
    T_97[20] <= UInt<1>("h00")
    T_97[21] <= UInt<1>("h00")
    T_97[22] <= UInt<1>("h00")
    T_97[23] <= UInt<1>("h00")
    T_97[24] <= UInt<1>("h00")
    T_97[25] <= UInt<1>("h00")
    T_97[26] <= UInt<1>("h00")
    T_97[27] <= UInt<1>("h00")
    T_97[28] <= UInt<1>("h00")
    T_97[29] <= UInt<1>("h00")
    T_97[30] <= UInt<1>("h00")
    T_97[31] <= UInt<1>("h00")
    T_97[32] <= UInt<1>("h01")
    T_97[33] <= UInt<1>("h01")
    T_97[34] <= UInt<1>("h01")
    T_97[35] <= UInt<1>("h01")
    T_97[36] <= UInt<1>("h01")
    T_97[37] <= UInt<1>("h01")
    T_97[38] <= UInt<1>("h01")
    T_97[39] <= UInt<1>("h01")
    T_97[40] <= UInt<1>("h01")
    T_97[41] <= UInt<1>("h01")
    T_97[42] <= UInt<1>("h01")
    T_97[43] <= UInt<1>("h01")
    T_97[44] <= UInt<1>("h01")
    T_97[45] <= UInt<1>("h01")
    T_97[46] <= UInt<1>("h01")
    T_97[47] <= UInt<1>("h01")
    T_97[48] <= UInt<1>("h01")
    T_97[49] <= UInt<1>("h01")
    T_97[50] <= UInt<1>("h01")
    T_97[51] <= UInt<1>("h01")
    T_97[52] <= UInt<1>("h01")
    T_97[53] <= UInt<1>("h00")
    reg T_154 : UInt<5>, clk with :
       reset => (reset, UInt<5>("h00"))
    wire T_182 : UInt<4>[26]
    T_182[0] <= UInt<1>("h00")
    T_182[1] <= UInt<1>("h01")
    T_182[2] <= UInt<2>("h02")
    T_182[3] <= UInt<2>("h03")
    T_182[4] <= UInt<1>("h00")
    T_182[5] <= UInt<1>("h01")
    T_182[6] <= UInt<2>("h02")
    T_182[7] <= UInt<2>("h03")
    T_182[8] <= UInt<3>("h04")
    T_182[9] <= UInt<3>("h05")
    T_182[10] <= UInt<3>("h06")
    T_182[11] <= UInt<3>("h07")
    T_182[12] <= UInt<4>("h08")
    T_182[13] <= UInt<4>("h09")
    T_182[14] <= UInt<4>("h0a")
    T_182[15] <= UInt<4>("h0b")
    T_182[16] <= UInt<4>("h0c")
    T_182[17] <= UInt<4>("h0d")
    T_182[18] <= UInt<4>("h0e")
    T_182[19] <= UInt<1>("h00")
    T_182[20] <= UInt<1>("h01")
    T_182[21] <= UInt<2>("h02")
    T_182[22] <= UInt<2>("h03")
    T_182[23] <= UInt<3>("h04")
    T_182[24] <= UInt<3>("h05")
    T_182[25] <= UInt<1>("h00")
    wire T_237 : UInt<31>[26]
    T_237[0] <= UInt<1>("h00")
    T_237[1] <= UInt<2>("h03")
    T_237[2] <= UInt<3>("h06")
    T_237[3] <= UInt<4>("h09")
    T_237[4] <= UInt<31>("h07dcd07ac")
    T_237[5] <= UInt<31>("h070890d84")
    T_237[6] <= UInt<26>("h02f45883")
    T_237[7] <= UInt<31>("h0787ada79")
    T_237[8] <= UInt<29>("h016866878")
    T_237[9] <= UInt<30>("h02331b107")
    T_237[10] <= UInt<30>("h0280e4938")
    T_237[11] <= UInt<29>("h0107fb3ac")
    T_237[12] <= UInt<30>("h02f19d47b")
    T_237[13] <= UInt<29>("h012c3d7cc")
    T_237[14] <= UInt<31>("h05a432a9c")
    T_237[15] <= UInt<26>("h02f9778f")
    T_237[16] <= UInt<22>("h02d705a")
    T_237[17] <= UInt<31>("h045fb9184")
    T_237[18] <= UInt<25>("h012e47af")
    T_237[19] <= UInt<31>("h07b744e21")
    T_237[20] <= UInt<31>("h0480ebc3d")
    T_237[21] <= UInt<28>("h0d5ff365")
    T_237[22] <= UInt<30>("h0205e7973")
    T_237[23] <= UInt<31>("h05004cbd2")
    T_237[24] <= UInt<30>("h024988736")
    T_237[25] <= UInt<1>("h00")
    device_under_test.io.in.bits.header <= T_182[T_154]
    device_under_test.io.in.bits.body <= T_237[T_154]
    device_under_test.io.in.valid <= T_97[T_19]
    node T_268 = and(device_under_test.io.in.valid, device_under_test.io.in.ready)
    when T_268 :
      node T_270 = eq(T_154, UInt<5>("h018"))
      node T_272 = and(UInt<1>("h01"), T_270)
      node T_275 = tail(add(T_154, UInt<1>("h01")),1)
      node T_276 = mux(T_272, UInt<1>("h00"), T_275)
      T_154 <= T_276
      node T_278 = eq(T_21, UInt<1>("h00"))
      when T_278 :
        node T_280 = eq(T_19, UInt<6>("h034"))
        when T_280 :
          T_21 <= UInt<1>("h01")
          skip
        node T_283 = tail(add(T_19, UInt<1>("h01")),1)
        T_19 <= T_283
        skip
      skip
    wire T_339 : UInt<1>[54]
    T_339[0] <= UInt<1>("h00")
    T_339[1] <= UInt<1>("h01")
    T_339[2] <= UInt<1>("h00")
    T_339[3] <= UInt<1>("h01")
    T_339[4] <= UInt<1>("h00")
    T_339[5] <= UInt<1>("h01")
    T_339[6] <= UInt<1>("h00")
    T_339[7] <= UInt<1>("h01")
    T_339[8] <= UInt<1>("h00")
    T_339[9] <= UInt<1>("h00")
    T_339[10] <= UInt<1>("h00")
    T_339[11] <= UInt<1>("h00")
    T_339[12] <= UInt<1>("h00")
    T_339[13] <= UInt<1>("h00")
    T_339[14] <= UInt<1>("h00")
    T_339[15] <= UInt<1>("h00")
    T_339[16] <= UInt<1>("h00")
    T_339[17] <= UInt<1>("h01")
    T_339[18] <= UInt<1>("h01")
    T_339[19] <= UInt<1>("h01")
    T_339[20] <= UInt<1>("h01")
    T_339[21] <= UInt<1>("h01")
    T_339[22] <= UInt<1>("h01")
    T_339[23] <= UInt<1>("h01")
    T_339[24] <= UInt<1>("h01")
    T_339[25] <= UInt<1>("h01")
    T_339[26] <= UInt<1>("h01")
    T_339[27] <= UInt<1>("h01")
    T_339[28] <= UInt<1>("h01")
    T_339[29] <= UInt<1>("h01")
    T_339[30] <= UInt<1>("h01")
    T_339[31] <= UInt<1>("h01")
    T_339[32] <= UInt<1>("h00")
    T_339[33] <= UInt<1>("h00")
    T_339[34] <= UInt<1>("h00")
    T_339[35] <= UInt<1>("h00")
    T_339[36] <= UInt<1>("h00")
    T_339[37] <= UInt<1>("h00")
    T_339[38] <= UInt<1>("h00")
    T_339[39] <= UInt<1>("h00")
    T_339[40] <= UInt<1>("h00")
    T_339[41] <= UInt<1>("h00")
    T_339[42] <= UInt<1>("h00")
    T_339[43] <= UInt<1>("h00")
    T_339[44] <= UInt<1>("h00")
    T_339[45] <= UInt<1>("h00")
    T_339[46] <= UInt<1>("h00")
    T_339[47] <= UInt<1>("h00")
    T_339[48] <= UInt<1>("h00")
    T_339[49] <= UInt<1>("h00")
    T_339[50] <= UInt<1>("h00")
    T_339[51] <= UInt<1>("h00")
    T_339[52] <= UInt<1>("h00")
    T_339[53] <= UInt<1>("h00")
    reg T_396 : UInt<5>, clk with :
       reset => (reset, UInt<5>("h00"))
    wire T_418 : UInt<2>[20]
    T_418[0] <= UInt<1>("h01")
    T_418[1] <= UInt<2>("h02")
    T_418[2] <= UInt<2>("h03")
    T_418[3] <= UInt<1>("h00")
    T_418[4] <= UInt<2>("h02")
    T_418[5] <= UInt<2>("h03")
    T_418[6] <= UInt<1>("h00")
    T_418[7] <= UInt<2>("h02")
    T_418[8] <= UInt<2>("h02")
    T_418[9] <= UInt<1>("h01")
    T_418[10] <= UInt<2>("h02")
    T_418[11] <= UInt<1>("h00")
    T_418[12] <= UInt<2>("h02")
    T_418[13] <= UInt<2>("h03")
    T_418[14] <= UInt<1>("h01")
    T_418[15] <= UInt<1>("h01")
    T_418[16] <= UInt<1>("h01")
    T_418[17] <= UInt<2>("h02")
    T_418[18] <= UInt<2>("h03")
    T_418[19] <= UInt<1>("h00")
    wire T_461 : UInt<4>[20]
    T_461[0] <= UInt<1>("h00")
    T_461[1] <= UInt<1>("h01")
    T_461[2] <= UInt<2>("h02")
    T_461[3] <= UInt<2>("h03")
    T_461[4] <= UInt<1>("h00")
    T_461[5] <= UInt<1>("h01")
    T_461[6] <= UInt<2>("h02")
    T_461[7] <= UInt<2>("h03")
    T_461[8] <= UInt<3>("h04")
    T_461[9] <= UInt<3>("h05")
    T_461[10] <= UInt<3>("h06")
    T_461[11] <= UInt<3>("h07")
    T_461[12] <= UInt<4>("h08")
    T_461[13] <= UInt<4>("h09")
    T_461[14] <= UInt<4>("h0a")
    T_461[15] <= UInt<4>("h0b")
    T_461[16] <= UInt<4>("h0c")
    T_461[17] <= UInt<4>("h0d")
    T_461[18] <= UInt<4>("h0e")
    T_461[19] <= UInt<1>("h00")
    device_under_test.io.load_routing_table_request.bits.data <= T_418[T_396]
    device_under_test.io.load_routing_table_request.bits.addr <= T_461[T_396]
    device_under_test.io.load_routing_table_request.valid <= T_339[T_19]
    node T_486 = and(device_under_test.io.load_routing_table_request.valid, device_under_test.io.load_routing_table_request.ready)
    when T_486 :
      node T_488 = eq(T_396, UInt<5>("h012"))
      node T_490 = and(UInt<1>("h01"), T_488)
      node T_493 = tail(add(T_396, UInt<1>("h01")),1)
      node T_494 = mux(T_490, UInt<1>("h00"), T_493)
      T_396 <= T_494
      node T_496 = eq(T_21, UInt<1>("h00"))
      when T_496 :
        node T_498 = eq(T_19, UInt<6>("h034"))
        when T_498 :
          T_21 <= UInt<1>("h01")
          skip
        node T_501 = tail(add(T_19, UInt<1>("h01")),1)
        T_19 <= T_501
        skip
      skip
    wire T_557 : UInt<1>[54]
    T_557[0] <= UInt<1>("h01")
    T_557[1] <= UInt<1>("h00")
    T_557[2] <= UInt<1>("h01")
    T_557[3] <= UInt<1>("h00")
    T_557[4] <= UInt<1>("h01")
    T_557[5] <= UInt<1>("h00")
    T_557[6] <= UInt<1>("h01")
    T_557[7] <= UInt<1>("h00")
    T_557[8] <= UInt<1>("h01")
    T_557[9] <= UInt<1>("h01")
    T_557[10] <= UInt<1>("h01")
    T_557[11] <= UInt<1>("h01")
    T_557[12] <= UInt<1>("h01")
    T_557[13] <= UInt<1>("h00")
    T_557[14] <= UInt<1>("h00")
    T_557[15] <= UInt<1>("h00")
    T_557[16] <= UInt<1>("h00")
    T_557[17] <= UInt<1>("h00")
    T_557[18] <= UInt<1>("h00")
    T_557[19] <= UInt<1>("h00")
    T_557[20] <= UInt<1>("h00")
    T_557[21] <= UInt<1>("h00")
    T_557[22] <= UInt<1>("h00")
    T_557[23] <= UInt<1>("h00")
    T_557[24] <= UInt<1>("h00")
    T_557[25] <= UInt<1>("h00")
    T_557[26] <= UInt<1>("h00")
    T_557[27] <= UInt<1>("h00")
    T_557[28] <= UInt<1>("h00")
    T_557[29] <= UInt<1>("h00")
    T_557[30] <= UInt<1>("h00")
    T_557[31] <= UInt<1>("h00")
    T_557[32] <= UInt<1>("h00")
    T_557[33] <= UInt<1>("h00")
    T_557[34] <= UInt<1>("h00")
    T_557[35] <= UInt<1>("h00")
    T_557[36] <= UInt<1>("h00")
    T_557[37] <= UInt<1>("h00")
    T_557[38] <= UInt<1>("h00")
    T_557[39] <= UInt<1>("h00")
    T_557[40] <= UInt<1>("h00")
    T_557[41] <= UInt<1>("h00")
    T_557[42] <= UInt<1>("h00")
    T_557[43] <= UInt<1>("h00")
    T_557[44] <= UInt<1>("h00")
    T_557[45] <= UInt<1>("h00")
    T_557[46] <= UInt<1>("h00")
    T_557[47] <= UInt<1>("h00")
    T_557[48] <= UInt<1>("h00")
    T_557[49] <= UInt<1>("h00")
    T_557[50] <= UInt<1>("h00")
    T_557[51] <= UInt<1>("h00")
    T_557[52] <= UInt<1>("h00")
    T_557[53] <= UInt<1>("h00")
    reg T_614 : UInt<4>, clk with :
       reset => (reset, UInt<4>("h00"))
    wire T_626 : UInt<2>[10]
    T_626[0] <= UInt<1>("h00")
    T_626[1] <= UInt<1>("h00")
    T_626[2] <= UInt<1>("h01")
    T_626[3] <= UInt<2>("h02")
    T_626[4] <= UInt<2>("h03")
    T_626[5] <= UInt<2>("h03")
    T_626[6] <= UInt<2>("h02")
    T_626[7] <= UInt<1>("h01")
    T_626[8] <= UInt<1>("h00")
    T_626[9] <= UInt<1>("h00")
    device_under_test.io.read_routing_table_request.bits.addr <= T_626[T_614]
    device_under_test.io.read_routing_table_request.valid <= T_557[T_19]
    node T_640 = and(device_under_test.io.read_routing_table_request.valid, device_under_test.io.read_routing_table_request.ready)
    when T_640 :
      node T_642 = eq(T_614, UInt<4>("h08"))
      node T_644 = and(UInt<1>("h01"), T_642)
      node T_647 = tail(add(T_614, UInt<1>("h01")),1)
      node T_648 = mux(T_644, UInt<1>("h00"), T_647)
      T_614 <= T_648
      node T_650 = eq(T_21, UInt<1>("h00"))
      when T_650 :
        node T_652 = eq(T_19, UInt<6>("h034"))
        when T_652 :
          T_21 <= UInt<1>("h01")
          skip
        node T_655 = tail(add(T_19, UInt<1>("h01")),1)
        T_19 <= T_655
        skip
      skip
    wire T_711 : UInt<1>[54]
    T_711[0] <= UInt<1>("h00")
    T_711[1] <= UInt<1>("h00")
    T_711[2] <= UInt<1>("h00")
    T_711[3] <= UInt<1>("h00")
    T_711[4] <= UInt<1>("h00")
    T_711[5] <= UInt<1>("h00")
    T_711[6] <= UInt<1>("h00")
    T_711[7] <= UInt<1>("h00")
    T_711[8] <= UInt<1>("h00")
    T_711[9] <= UInt<1>("h00")
    T_711[10] <= UInt<1>("h00")
    T_711[11] <= UInt<1>("h00")
    T_711[12] <= UInt<1>("h01")
    T_711[13] <= UInt<1>("h00")
    T_711[14] <= UInt<1>("h00")
    T_711[15] <= UInt<1>("h01")
    T_711[16] <= UInt<1>("h00")
    T_711[17] <= UInt<1>("h00")
    T_711[18] <= UInt<1>("h00")
    T_711[19] <= UInt<1>("h00")
    T_711[20] <= UInt<1>("h01")
    T_711[21] <= UInt<1>("h00")
    T_711[22] <= UInt<1>("h00")
    T_711[23] <= UInt<1>("h00")
    T_711[24] <= UInt<1>("h00")
    T_711[25] <= UInt<1>("h00")
    T_711[26] <= UInt<1>("h00")
    T_711[27] <= UInt<1>("h00")
    T_711[28] <= UInt<1>("h00")
    T_711[29] <= UInt<1>("h00")
    T_711[30] <= UInt<1>("h01")
    T_711[31] <= UInt<1>("h00")
    T_711[32] <= UInt<1>("h00")
    T_711[33] <= UInt<1>("h00")
    T_711[34] <= UInt<1>("h00")
    T_711[35] <= UInt<1>("h00")
    T_711[36] <= UInt<1>("h00")
    T_711[37] <= UInt<1>("h00")
    T_711[38] <= UInt<1>("h00")
    T_711[39] <= UInt<1>("h00")
    T_711[40] <= UInt<1>("h00")
    T_711[41] <= UInt<1>("h00")
    T_711[42] <= UInt<1>("h00")
    T_711[43] <= UInt<1>("h00")
    T_711[44] <= UInt<1>("h00")
    T_711[45] <= UInt<1>("h00")
    T_711[46] <= UInt<1>("h00")
    T_711[47] <= UInt<1>("h00")
    T_711[48] <= UInt<1>("h00")
    T_711[49] <= UInt<1>("h00")
    T_711[50] <= UInt<1>("h00")
    T_711[51] <= UInt<1>("h00")
    T_711[52] <= UInt<1>("h00")
    T_711[53] <= UInt<1>("h00")
    reg T_768 : UInt<6>, clk with :
       reset => (reset, UInt<6>("h00"))
    wire T_775 : UInt<29>[5]
    T_775[0] <= UInt<4>("h09")
    T_775[1] <= UInt<26>("h02f45883")
    T_775[2] <= UInt<29>("h0107fb3ac")
    T_775[3] <= UInt<28>("h0d5ff365")
    T_775[4] <= UInt<1>("h00")
    device_under_test.io.outs[0].ready <= T_711[T_23]
    node T_783 = and(device_under_test.io.outs[0].ready, device_under_test.io.outs[0].valid)
    when T_783 :
      node T_786 = eq(reset, UInt<1>("h00"))
      when T_786 :
        printf(clk, UInt<1>(1), "output test event %d testing outs(0).bits.body = %d, should be %d
", T_23, device_under_test.io.outs[0].bits.body, T_775[T_768])
        skip
      node T_788 = neq(device_under_test.io.outs[0].bits.body, T_775[T_768])
      when T_788 :
        node T_791 = eq(reset, UInt<1>("h00"))
        when T_791 :
          printf(clk, UInt<1>(1), "Error: event %d outs(0).bits.body was %d should be %d
", T_23, device_under_test.io.outs[0].bits.body, T_775[T_768])
          skip
        node T_794 = eq(reset, UInt<1>("h00"))
        when T_794 :
          node T_796 = eq(UInt<1>("h00"), UInt<1>("h00"))
          when T_796 :
            node T_798 = eq(reset, UInt<1>("h00"))
            when T_798 :
              printf(clk, UInt<1>(1), "Assertion failed: (TODO: code / lineno)")
              skip
            stop(clk, UInt<1>(1), 1)
            skip
          skip
        node T_800 = eq(reset, UInt<1>("h00"))
        when T_800 :
          stop(clk, UInt<1>(1), 0)
          skip
        skip
      node T_802 = eq(T_768, UInt<6>("h021"))
      node T_804 = and(UInt<1>("h01"), T_802)
      node T_807 = tail(add(T_768, UInt<1>("h01")),1)
      node T_808 = mux(T_804, UInt<1>("h00"), T_807)
      T_768 <= T_808
      node T_810 = eq(T_25, UInt<1>("h00"))
      when T_810 :
        node T_812 = eq(T_23, UInt<6>("h021"))
        when T_812 :
          T_25 <= UInt<1>("h01")
          skip
        node T_815 = tail(add(T_23, UInt<1>("h01")),1)
        T_23 <= T_815
        skip
      skip
    wire T_871 : UInt<1>[54]
    T_871[0] <= UInt<1>("h00")
    T_871[1] <= UInt<1>("h00")
    T_871[2] <= UInt<1>("h00")
    T_871[3] <= UInt<1>("h00")
    T_871[4] <= UInt<1>("h00")
    T_871[5] <= UInt<1>("h00")
    T_871[6] <= UInt<1>("h00")
    T_871[7] <= UInt<1>("h00")
    T_871[8] <= UInt<1>("h00")
    T_871[9] <= UInt<1>("h00")
    T_871[10] <= UInt<1>("h00")
    T_871[11] <= UInt<1>("h01")
    T_871[12] <= UInt<1>("h00")
    T_871[13] <= UInt<1>("h00")
    T_871[14] <= UInt<1>("h01")
    T_871[15] <= UInt<1>("h00")
    T_871[16] <= UInt<1>("h00")
    T_871[17] <= UInt<1>("h00")
    T_871[18] <= UInt<1>("h00")
    T_871[19] <= UInt<1>("h00")
    T_871[20] <= UInt<1>("h00")
    T_871[21] <= UInt<1>("h00")
    T_871[22] <= UInt<1>("h01")
    T_871[23] <= UInt<1>("h00")
    T_871[24] <= UInt<1>("h00")
    T_871[25] <= UInt<1>("h00")
    T_871[26] <= UInt<1>("h00")
    T_871[27] <= UInt<1>("h01")
    T_871[28] <= UInt<1>("h00")
    T_871[29] <= UInt<1>("h01")
    T_871[30] <= UInt<1>("h00")
    T_871[31] <= UInt<1>("h00")
    T_871[32] <= UInt<1>("h00")
    T_871[33] <= UInt<1>("h00")
    T_871[34] <= UInt<1>("h00")
    T_871[35] <= UInt<1>("h00")
    T_871[36] <= UInt<1>("h00")
    T_871[37] <= UInt<1>("h00")
    T_871[38] <= UInt<1>("h00")
    T_871[39] <= UInt<1>("h00")
    T_871[40] <= UInt<1>("h00")
    T_871[41] <= UInt<1>("h00")
    T_871[42] <= UInt<1>("h00")
    T_871[43] <= UInt<1>("h00")
    T_871[44] <= UInt<1>("h00")
    T_871[45] <= UInt<1>("h00")
    T_871[46] <= UInt<1>("h00")
    T_871[47] <= UInt<1>("h00")
    T_871[48] <= UInt<1>("h00")
    T_871[49] <= UInt<1>("h00")
    T_871[50] <= UInt<1>("h00")
    T_871[51] <= UInt<1>("h00")
    T_871[52] <= UInt<1>("h00")
    T_871[53] <= UInt<1>("h00")
    reg T_928 : UInt<6>, clk with :
       reset => (reset, UInt<6>("h00"))
    wire T_936 : UInt<31>[6]
    T_936[0] <= UInt<3>("h06")
    T_936[1] <= UInt<31>("h070890d84")
    T_936[2] <= UInt<29>("h012c3d7cc")
    T_936[3] <= UInt<25>("h012e47af")
    T_936[4] <= UInt<31>("h0480ebc3d")
    T_936[5] <= UInt<1>("h00")
    device_under_test.io.outs[3].ready <= T_871[T_23]
    node T_945 = and(device_under_test.io.outs[3].ready, device_under_test.io.outs[3].valid)
    when T_945 :
      node T_948 = eq(reset, UInt<1>("h00"))
      when T_948 :
        printf(clk, UInt<1>(1), "output test event %d testing outs(3).bits.body = %d, should be %d
", T_23, device_under_test.io.outs[3].bits.body, T_936[T_928])
        skip
      node T_950 = neq(device_under_test.io.outs[3].bits.body, T_936[T_928])
      when T_950 :
        node T_953 = eq(reset, UInt<1>("h00"))
        when T_953 :
          printf(clk, UInt<1>(1), "Error: event %d outs(3).bits.body was %d should be %d
", T_23, device_under_test.io.outs[3].bits.body, T_936[T_928])
          skip
        node T_956 = eq(reset, UInt<1>("h00"))
        when T_956 :
          node T_958 = eq(UInt<1>("h00"), UInt<1>("h00"))
          when T_958 :
            node T_960 = eq(reset, UInt<1>("h00"))
            when T_960 :
              printf(clk, UInt<1>(1), "Assertion failed: (TODO: code / lineno)")
              skip
            stop(clk, UInt<1>(1), 1)
            skip
          skip
        node T_962 = eq(reset, UInt<1>("h00"))
        when T_962 :
          stop(clk, UInt<1>(1), 0)
          skip
        skip
      node T_964 = eq(T_928, UInt<6>("h021"))
      node T_966 = and(UInt<1>("h01"), T_964)
      node T_969 = tail(add(T_928, UInt<1>("h01")),1)
      node T_970 = mux(T_966, UInt<1>("h00"), T_969)
      T_928 <= T_970
      node T_972 = eq(T_25, UInt<1>("h00"))
      when T_972 :
        node T_974 = eq(T_23, UInt<6>("h021"))
        when T_974 :
          T_25 <= UInt<1>("h01")
          skip
        node T_977 = tail(add(T_23, UInt<1>("h01")),1)
        T_23 <= T_977
        skip
      skip
    wire T_1033 : UInt<1>[54]
    T_1033[0] <= UInt<1>("h00")
    T_1033[1] <= UInt<1>("h00")
    T_1033[2] <= UInt<1>("h00")
    T_1033[3] <= UInt<1>("h00")
    T_1033[4] <= UInt<1>("h00")
    T_1033[5] <= UInt<1>("h00")
    T_1033[6] <= UInt<1>("h00")
    T_1033[7] <= UInt<1>("h00")
    T_1033[8] <= UInt<1>("h00")
    T_1033[9] <= UInt<1>("h01")
    T_1033[10] <= UInt<1>("h00")
    T_1033[11] <= UInt<1>("h00")
    T_1033[12] <= UInt<1>("h00")
    T_1033[13] <= UInt<1>("h00")
    T_1033[14] <= UInt<1>("h00")
    T_1033[15] <= UInt<1>("h00")
    T_1033[16] <= UInt<1>("h00")
    T_1033[17] <= UInt<1>("h00")
    T_1033[18] <= UInt<1>("h01")
    T_1033[19] <= UInt<1>("h00")
    T_1033[20] <= UInt<1>("h00")
    T_1033[21] <= UInt<1>("h00")
    T_1033[22] <= UInt<1>("h00")
    T_1033[23] <= UInt<1>("h01")
    T_1033[24] <= UInt<1>("h01")
    T_1033[25] <= UInt<1>("h01")
    T_1033[26] <= UInt<1>("h00")
    T_1033[27] <= UInt<1>("h00")
    T_1033[28] <= UInt<1>("h00")
    T_1033[29] <= UInt<1>("h00")
    T_1033[30] <= UInt<1>("h00")
    T_1033[31] <= UInt<1>("h00")
    T_1033[32] <= UInt<1>("h00")
    T_1033[33] <= UInt<1>("h01")
    T_1033[34] <= UInt<1>("h00")
    T_1033[35] <= UInt<1>("h00")
    T_1033[36] <= UInt<1>("h00")
    T_1033[37] <= UInt<1>("h00")
    T_1033[38] <= UInt<1>("h00")
    T_1033[39] <= UInt<1>("h00")
    T_1033[40] <= UInt<1>("h00")
    T_1033[41] <= UInt<1>("h00")
    T_1033[42] <= UInt<1>("h00")
    T_1033[43] <= UInt<1>("h00")
    T_1033[44] <= UInt<1>("h00")
    T_1033[45] <= UInt<1>("h00")
    T_1033[46] <= UInt<1>("h00")
    T_1033[47] <= UInt<1>("h00")
    T_1033[48] <= UInt<1>("h00")
    T_1033[49] <= UInt<1>("h00")
    T_1033[50] <= UInt<1>("h00")
    T_1033[51] <= UInt<1>("h00")
    T_1033[52] <= UInt<1>("h00")
    T_1033[53] <= UInt<1>("h00")
    reg T_1090 : UInt<6>, clk with :
       reset => (reset, UInt<6>("h00"))
    wire T_1099 : UInt<31>[7]
    T_1099[0] <= UInt<1>("h00")
    T_1099[1] <= UInt<30>("h02331b107")
    T_1099[2] <= UInt<31>("h05a432a9c")
    T_1099[3] <= UInt<26>("h02f9778f")
    T_1099[4] <= UInt<22>("h02d705a")
    T_1099[5] <= UInt<30>("h024988736")
    T_1099[6] <= UInt<1>("h00")
    device_under_test.io.outs[1].ready <= T_1033[T_23]
    node T_1109 = and(device_under_test.io.outs[1].ready, device_under_test.io.outs[1].valid)
    when T_1109 :
      node T_1112 = eq(reset, UInt<1>("h00"))
      when T_1112 :
        printf(clk, UInt<1>(1), "output test event %d testing outs(1).bits.body = %d, should be %d
", T_23, device_under_test.io.outs[1].bits.body, T_1099[T_1090])
        skip
      node T_1114 = neq(device_under_test.io.outs[1].bits.body, T_1099[T_1090])
      when T_1114 :
        node T_1117 = eq(reset, UInt<1>("h00"))
        when T_1117 :
          printf(clk, UInt<1>(1), "Error: event %d outs(1).bits.body was %d should be %d
", T_23, device_under_test.io.outs[1].bits.body, T_1099[T_1090])
          skip
        node T_1120 = eq(reset, UInt<1>("h00"))
        when T_1120 :
          node T_1122 = eq(UInt<1>("h00"), UInt<1>("h00"))
          when T_1122 :
            node T_1124 = eq(reset, UInt<1>("h00"))
            when T_1124 :
              printf(clk, UInt<1>(1), "Assertion failed: (TODO: code / lineno)")
              skip
            stop(clk, UInt<1>(1), 1)
            skip
          skip
        node T_1126 = eq(reset, UInt<1>("h00"))
        when T_1126 :
          stop(clk, UInt<1>(1), 0)
          skip
        skip
      node T_1128 = eq(T_1090, UInt<6>("h021"))
      node T_1130 = and(UInt<1>("h01"), T_1128)
      node T_1133 = tail(add(T_1090, UInt<1>("h01")),1)
      node T_1134 = mux(T_1130, UInt<1>("h00"), T_1133)
      T_1090 <= T_1134
      node T_1136 = eq(T_25, UInt<1>("h00"))
      when T_1136 :
        node T_1138 = eq(T_23, UInt<6>("h021"))
        when T_1138 :
          T_25 <= UInt<1>("h01")
          skip
        node T_1141 = tail(add(T_23, UInt<1>("h01")),1)
        T_23 <= T_1141
        skip
      skip
    wire T_1197 : UInt<1>[54]
    T_1197[0] <= UInt<1>("h00")
    T_1197[1] <= UInt<1>("h00")
    T_1197[2] <= UInt<1>("h00")
    T_1197[3] <= UInt<1>("h00")
    T_1197[4] <= UInt<1>("h00")
    T_1197[5] <= UInt<1>("h00")
    T_1197[6] <= UInt<1>("h00")
    T_1197[7] <= UInt<1>("h00")
    T_1197[8] <= UInt<1>("h00")
    T_1197[9] <= UInt<1>("h00")
    T_1197[10] <= UInt<1>("h01")
    T_1197[11] <= UInt<1>("h00")
    T_1197[12] <= UInt<1>("h00")
    T_1197[13] <= UInt<1>("h01")
    T_1197[14] <= UInt<1>("h00")
    T_1197[15] <= UInt<1>("h00")
    T_1197[16] <= UInt<1>("h01")
    T_1197[17] <= UInt<1>("h01")
    T_1197[18] <= UInt<1>("h00")
    T_1197[19] <= UInt<1>("h01")
    T_1197[20] <= UInt<1>("h00")
    T_1197[21] <= UInt<1>("h01")
    T_1197[22] <= UInt<1>("h00")
    T_1197[23] <= UInt<1>("h00")
    T_1197[24] <= UInt<1>("h00")
    T_1197[25] <= UInt<1>("h00")
    T_1197[26] <= UInt<1>("h01")
    T_1197[27] <= UInt<1>("h00")
    T_1197[28] <= UInt<1>("h01")
    T_1197[29] <= UInt<1>("h00")
    T_1197[30] <= UInt<1>("h00")
    T_1197[31] <= UInt<1>("h01")
    T_1197[32] <= UInt<1>("h01")
    T_1197[33] <= UInt<1>("h00")
    T_1197[34] <= UInt<1>("h00")
    T_1197[35] <= UInt<1>("h00")
    T_1197[36] <= UInt<1>("h00")
    T_1197[37] <= UInt<1>("h00")
    T_1197[38] <= UInt<1>("h00")
    T_1197[39] <= UInt<1>("h00")
    T_1197[40] <= UInt<1>("h00")
    T_1197[41] <= UInt<1>("h00")
    T_1197[42] <= UInt<1>("h00")
    T_1197[43] <= UInt<1>("h00")
    T_1197[44] <= UInt<1>("h00")
    T_1197[45] <= UInt<1>("h00")
    T_1197[46] <= UInt<1>("h00")
    T_1197[47] <= UInt<1>("h00")
    T_1197[48] <= UInt<1>("h00")
    T_1197[49] <= UInt<1>("h00")
    T_1197[50] <= UInt<1>("h00")
    T_1197[51] <= UInt<1>("h00")
    T_1197[52] <= UInt<1>("h00")
    T_1197[53] <= UInt<1>("h00")
    reg T_1254 : UInt<6>, clk with :
       reset => (reset, UInt<6>("h00"))
    wire T_1267 : UInt<31>[11]
    T_1267[0] <= UInt<2>("h03")
    T_1267[1] <= UInt<31>("h07dcd07ac")
    T_1267[2] <= UInt<31>("h0787ada79")
    T_1267[3] <= UInt<29>("h016866878")
    T_1267[4] <= UInt<30>("h0280e4938")
    T_1267[5] <= UInt<30>("h02f19d47b")
    T_1267[6] <= UInt<31>("h045fb9184")
    T_1267[7] <= UInt<31>("h07b744e21")
    T_1267[8] <= UInt<30>("h0205e7973")
    T_1267[9] <= UInt<31>("h05004cbd2")
    T_1267[10] <= UInt<1>("h00")
    device_under_test.io.outs[2].ready <= T_1197[T_23]
    node T_1281 = and(device_under_test.io.outs[2].ready, device_under_test.io.outs[2].valid)
    when T_1281 :
      node T_1284 = eq(reset, UInt<1>("h00"))
      when T_1284 :
        printf(clk, UInt<1>(1), "output test event %d testing outs(2).bits.body = %d, should be %d
", T_23, device_under_test.io.outs[2].bits.body, T_1267[T_1254])
        skip
      node T_1286 = neq(device_under_test.io.outs[2].bits.body, T_1267[T_1254])
      when T_1286 :
        node T_1289 = eq(reset, UInt<1>("h00"))
        when T_1289 :
          printf(clk, UInt<1>(1), "Error: event %d outs(2).bits.body was %d should be %d
", T_23, device_under_test.io.outs[2].bits.body, T_1267[T_1254])
          skip
        node T_1292 = eq(reset, UInt<1>("h00"))
        when T_1292 :
          node T_1294 = eq(UInt<1>("h00"), UInt<1>("h00"))
          when T_1294 :
            node T_1296 = eq(reset, UInt<1>("h00"))
            when T_1296 :
              printf(clk, UInt<1>(1), "Assertion failed: (TODO: code / lineno)")
              skip
            stop(clk, UInt<1>(1), 1)
            skip
          skip
        node T_1298 = eq(reset, UInt<1>("h00"))
        when T_1298 :
          stop(clk, UInt<1>(1), 0)
          skip
        skip
      node T_1300 = eq(T_1254, UInt<6>("h021"))
      node T_1302 = and(UInt<1>("h01"), T_1300)
      node T_1305 = tail(add(T_1254, UInt<1>("h01")),1)
      node T_1306 = mux(T_1302, UInt<1>("h00"), T_1305)
      T_1254 <= T_1306
      node T_1308 = eq(T_25, UInt<1>("h00"))
      when T_1308 :
        node T_1310 = eq(T_23, UInt<6>("h021"))
        when T_1310 :
          T_25 <= UInt<1>("h01")
          skip
        node T_1313 = tail(add(T_23, UInt<1>("h01")),1)
        T_23 <= T_1313
        skip
      skip
    wire T_1369 : UInt<1>[54]
    T_1369[0] <= UInt<1>("h01")
    T_1369[1] <= UInt<1>("h01")
    T_1369[2] <= UInt<1>("h01")
    T_1369[3] <= UInt<1>("h01")
    T_1369[4] <= UInt<1>("h01")
    T_1369[5] <= UInt<1>("h01")
    T_1369[6] <= UInt<1>("h01")
    T_1369[7] <= UInt<1>("h01")
    T_1369[8] <= UInt<1>("h01")
    T_1369[9] <= UInt<1>("h00")
    T_1369[10] <= UInt<1>("h00")
    T_1369[11] <= UInt<1>("h00")
    T_1369[12] <= UInt<1>("h00")
    T_1369[13] <= UInt<1>("h00")
    T_1369[14] <= UInt<1>("h00")
    T_1369[15] <= UInt<1>("h00")
    T_1369[16] <= UInt<1>("h00")
    T_1369[17] <= UInt<1>("h00")
    T_1369[18] <= UInt<1>("h00")
    T_1369[19] <= UInt<1>("h00")
    T_1369[20] <= UInt<1>("h00")
    T_1369[21] <= UInt<1>("h00")
    T_1369[22] <= UInt<1>("h00")
    T_1369[23] <= UInt<1>("h00")
    T_1369[24] <= UInt<1>("h00")
    T_1369[25] <= UInt<1>("h00")
    T_1369[26] <= UInt<1>("h00")
    T_1369[27] <= UInt<1>("h00")
    T_1369[28] <= UInt<1>("h00")
    T_1369[29] <= UInt<1>("h00")
    T_1369[30] <= UInt<1>("h00")
    T_1369[31] <= UInt<1>("h00")
    T_1369[32] <= UInt<1>("h00")
    T_1369[33] <= UInt<1>("h00")
    T_1369[34] <= UInt<1>("h00")
    T_1369[35] <= UInt<1>("h00")
    T_1369[36] <= UInt<1>("h00")
    T_1369[37] <= UInt<1>("h00")
    T_1369[38] <= UInt<1>("h00")
    T_1369[39] <= UInt<1>("h00")
    T_1369[40] <= UInt<1>("h00")
    T_1369[41] <= UInt<1>("h00")
    T_1369[42] <= UInt<1>("h00")
    T_1369[43] <= UInt<1>("h00")
    T_1369[44] <= UInt<1>("h00")
    T_1369[45] <= UInt<1>("h00")
    T_1369[46] <= UInt<1>("h00")
    T_1369[47] <= UInt<1>("h00")
    T_1369[48] <= UInt<1>("h00")
    T_1369[49] <= UInt<1>("h00")
    T_1369[50] <= UInt<1>("h00")
    T_1369[51] <= UInt<1>("h00")
    T_1369[52] <= UInt<1>("h00")
    T_1369[53] <= UInt<1>("h00")
    reg T_1426 : UInt<6>, clk with :
       reset => (reset, UInt<6>("h00"))
    wire T_1438 : UInt<2>[10]
    T_1438[0] <= UInt<1>("h00")
    T_1438[1] <= UInt<1>("h01")
    T_1438[2] <= UInt<2>("h02")
    T_1438[3] <= UInt<2>("h03")
    T_1438[4] <= UInt<1>("h00")
    T_1438[5] <= UInt<1>("h00")
    T_1438[6] <= UInt<2>("h03")
    T_1438[7] <= UInt<2>("h02")
    T_1438[8] <= UInt<1>("h01")
    T_1438[9] <= UInt<1>("h00")
    device_under_test.io.read_routing_table_response.ready <= T_1369[T_23]
    node T_1451 = and(device_under_test.io.read_routing_table_response.ready, device_under_test.io.read_routing_table_response.valid)
    when T_1451 :
      node T_1454 = eq(reset, UInt<1>("h00"))
      when T_1454 :
        printf(clk, UInt<1>(1), "output test event %d testing read_routing_table_response.bits = %d, should be %d
", T_23, device_under_test.io.read_routing_table_response.bits, T_1438[T_1426])
        skip
      node T_1456 = neq(device_under_test.io.read_routing_table_response.bits, T_1438[T_1426])
      when T_1456 :
        node T_1459 = eq(reset, UInt<1>("h00"))
        when T_1459 :
          printf(clk, UInt<1>(1), "Error: event %d read_routing_table_response.bits was %d should be %d
", T_23, device_under_test.io.read_routing_table_response.bits, T_1438[T_1426])
          skip
        node T_1462 = eq(reset, UInt<1>("h00"))
        when T_1462 :
          node T_1464 = eq(UInt<1>("h00"), UInt<1>("h00"))
          when T_1464 :
            node T_1466 = eq(reset, UInt<1>("h00"))
            when T_1466 :
              printf(clk, UInt<1>(1), "Assertion failed: (TODO: code / lineno)")
              skip
            stop(clk, UInt<1>(1), 1)
            skip
          skip
        node T_1468 = eq(reset, UInt<1>("h00"))
        when T_1468 :
          stop(clk, UInt<1>(1), 0)
          skip
        skip
      node T_1470 = eq(T_1426, UInt<6>("h021"))
      node T_1472 = and(UInt<1>("h01"), T_1470)
      node T_1475 = tail(add(T_1426, UInt<1>("h01")),1)
      node T_1476 = mux(T_1472, UInt<1>("h00"), T_1475)
      T_1426 <= T_1476
      node T_1478 = eq(T_25, UInt<1>("h00"))
      when T_1478 :
        node T_1480 = eq(T_23, UInt<6>("h021"))
        when T_1480 :
          T_25 <= UInt<1>("h01")
          skip
        node T_1483 = tail(add(T_23, UInt<1>("h01")),1)
        T_23 <= T_1483
        skip
      skip
    node T_1485 = eq(reset, UInt<1>("h00"))
    when T_1485 :
      printf(clk, UInt<1>(1), "in_event_counter %d, out_event_counter %d
", T_19, T_23)
      skip
    


