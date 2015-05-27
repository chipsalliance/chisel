; RUN: firrtl -i %s -o %s.flo -X flo -p ct | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Tile :
   module ALU :
      input A : UInt<32>
      input B : UInt<32>
      input alu_op : UInt<4>
      output out : UInt<32>
      output sum : UInt<32>
      node tmp1 = bits(B, 4, 0)
      node shamt2 = tmp1
      node tmp3 = add-wrap(A, B)
      node tmp4 = sub-wrap(A, B)
      node tmp5 = convert(A)
      node tmp6 = dshr(tmp5, shamt2)
      node tmp7 = as-UInt(tmp6)
      node tmp8 = dshr(A, shamt2)
      node tmp9 = dshl(A, shamt2)
      node tmp10 = bits(tmp9, 31, 0)
      node tmp11 = convert(A)
      node tmp12 = convert(B)
      node tmp13 = lt(tmp11, tmp12)
      node tmp14 = lt(A, B)
      node tmp15 = bit-and(A, B)
      node tmp16 = bit-or(A, B)
      node tmp17 = bit-xor(A, B)
      node tmp18 = eq(UInt<4>(10), alu_op)
      node tmp19 = mux(tmp18, A, B)
      node tmp20 = eq(UInt<4>(4), alu_op)
      node tmp21 = mux(tmp20, tmp17, tmp19)
      node tmp22 = eq(UInt<4>(3), alu_op)
      node tmp23 = mux(tmp22, tmp16, tmp21)
      node tmp24 = eq(UInt<4>(2), alu_op)
      node tmp25 = mux(tmp24, tmp15, tmp23)
      node tmp26 = eq(UInt<4>(7), alu_op)
      node tmp27 = mux(tmp26, tmp14, tmp25)
      node tmp28 = eq(UInt<4>(5), alu_op)
      node tmp29 = mux(tmp28, tmp13, tmp27)
      node tmp30 = eq(UInt<4>(6), alu_op)
      node tmp31 = mux(tmp30, tmp10, tmp29)
      node tmp32 = eq(UInt<4>(8), alu_op)
      node tmp33 = mux(tmp32, tmp8, tmp31)
      node tmp34 = eq(UInt<4>(9), alu_op)
      node tmp35 = mux(tmp34, tmp7, tmp33)
      node tmp36 = eq(UInt<4>(1), alu_op)
      node tmp37 = mux(tmp36, tmp4, tmp35)
      node tmp38 = eq(UInt<4>(0), alu_op)
      node tmp39 = mux(tmp38, tmp3, tmp37)
      node tmp40 = bits(tmp39, 31, 0)
      out := tmp40
      node tmp41 = bit(alu_op, 0)
      node tmp42 = sub-wrap(UInt<1>(0), B)
      node tmp43 = mux(tmp41, tmp42, B)
      node tmp44 = add-wrap(A, tmp43)
      sum := tmp44   
   module BrCond :
      input rs1 : UInt<32>
      input rs2 : UInt<32>
      input br_type : UInt<3>
      output taken : UInt<1>
      node tmp45 = eq(rs1, rs2)
      node eq46 = tmp45
      node tmp47 = bit-not(eq46)
      node neq48 = tmp47
      node tmp49 = convert(rs1)
      node tmp50 = convert(rs2)
      node tmp51 = lt(tmp49, tmp50)
      node lt52 = tmp51
      node tmp53 = bit-not(lt52)
      node ge54 = tmp53
      node tmp55 = lt(rs1, rs2)
      node ltu56 = tmp55
      node tmp57 = bit-not(ltu56)
      node geu58 = tmp57
      node tmp59 = eq(br_type, UInt<3>(2))
      node tmp60 = bit-and(tmp59, eq46)
      node tmp61 = eq(br_type, UInt<3>(6))
      node tmp62 = bit-and(tmp61, neq48)
      node tmp63 = eq(br_type, UInt<3>(1))
      node tmp64 = bit-and(tmp63, lt52)
      node tmp65 = eq(br_type, UInt<3>(5))
      node tmp66 = bit-and(tmp65, ge54)
      node tmp67 = eq(br_type, UInt<3>(0))
      node tmp68 = bit-and(tmp67, ltu56)
      node tmp69 = eq(br_type, UInt<3>(4))
      node tmp70 = bit-and(tmp69, geu58)
      node tmp71 = bit-or(tmp68, tmp70)
      node tmp72 = bit-or(tmp66, tmp71)
      node tmp73 = bit-or(tmp64, tmp72)
      node tmp74 = bit-or(tmp62, tmp73)
      node tmp75 = bit-or(tmp60, tmp74)
      taken := tmp75   
   module RegFile :
      input raddr1 : UInt<5>
      input raddr2 : UInt<5>
      input wen : UInt<1>
      input waddr : UInt<5>
      input wdata : UInt<32>
      output rdata1 : UInt<32>
      output rdata2 : UInt<32>
      mem regs : UInt<32>[32]
      node tmp76 = eq(raddr1, UInt<1>(0))
      node tmp77 = bit-not(tmp76)
      accessor a78 = regs[raddr1]
      node tmp79 = mux(tmp77, a78, UInt<1>(0))
      rdata1 := tmp79
      node tmp80 = eq(raddr2, UInt<1>(0))
      node tmp81 = bit-not(tmp80)
      accessor a82 = regs[raddr2]
      node tmp83 = mux(tmp81, a82, UInt<1>(0))
      rdata2 := tmp83
      node tmp84 = eq(waddr, UInt<1>(0))
      node tmp85 = bit-not(tmp84)
      node tmp86 = bit-and(wen, tmp85)
      when tmp86 :
         accessor a87 = regs[waddr]
         a87 := wdata   
   module ImmGenWire :
      input inst : UInt<32>
      input sel : UInt<3>
      output out : UInt<32>
      node tmp88 = bits(inst, 31, 20)
      node tmp89 = convert(tmp88)
      node Iimm90 = tmp89
      node tmp91 = bits(inst, 31, 25)
      node tmp92 = bits(inst, 11, 7)
      node tmp93 = cat(tmp91, tmp92)
      node tmp94 = convert(tmp93)
      node Simm95 = tmp94
      node tmp96 = bit(inst, 31)
      node tmp97 = bit(inst, 7)
      node tmp98 = bits(inst, 30, 25)
      node tmp99 = bits(inst, 11, 8)
      node tmp100 = cat(tmp96, tmp97)
      node tmp101 = cat(tmp100, tmp98)
      node tmp102 = cat(tmp101, tmp99)
      node tmp103 = cat(tmp102, UInt<1>(0))
      node tmp104 = convert(tmp103)
      node Bimm105 = tmp104
      node tmp106 = bits(inst, 31, 12)
      node tmp107 = cat(tmp106, UInt<12>(0))
      node tmp108 = convert(tmp107)
      node Uimm109 = tmp108
      node tmp110 = bit(inst, 31)
      node tmp111 = bits(inst, 19, 12)
      node tmp112 = bit(inst, 20)
      node tmp113 = bits(inst, 30, 25)
      node tmp114 = bits(inst, 24, 21)
      node tmp115 = cat(tmp110, tmp111)
      node tmp116 = cat(tmp115, tmp112)
      node tmp117 = cat(tmp116, tmp113)
      node tmp118 = cat(tmp117, tmp114)
      node tmp119 = cat(tmp118, UInt<1>(0))
      node tmp120 = convert(tmp119)
      node Jimm121 = tmp120
      node tmp122 = bits(inst, 19, 15)
      node tmp123 = pad(tmp122, 32)
      node tmp124 = convert(tmp123)
      node Zimm125 = tmp124
      node tmp126 = eq(UInt<3>(3), sel)
      node tmp127 = mux(tmp126, Jimm121, Zimm125)
      node tmp128 = eq(UInt<3>(2), sel)
      node tmp129 = mux(tmp128, Uimm109, tmp127)
      node tmp130 = eq(UInt<3>(4), sel)
      node tmp131 = mux(tmp130, Bimm105, tmp129)
      node tmp132 = eq(UInt<3>(1), sel)
      node tmp133 = mux(tmp132, Simm95, tmp131)
      node tmp134 = eq(UInt<3>(0), sel)
      node tmp135 = mux(tmp134, Iimm90, tmp133)
      node tmp136 = as-UInt(tmp135)
      out := tmp136   
   module CSR :
      input cmd : UInt<2>
      input addr : UInt<12>
      input src : UInt<32>
      output host : {flip hid : UInt<1>, tohost : UInt<32>, status : UInt<32>}
      output data : UInt<32>
      reg reg_tohost238 : UInt<32>
      on-reset reg_tohost238 := UInt<1>(0)
      reg reg_status239 : UInt<32>
      on-reset reg_status239 := UInt<1>(0)
      host.tohost := reg_tohost238
      host.status := reg_status239
      node tmp240 = eq(UInt<12>(1291), addr)
      node tmp241 = mux(tmp240, host.hid, UInt<1>(0))
      node tmp242 = eq(UInt<12>(1290), addr)
      node tmp243 = mux(tmp242, reg_status239, tmp241)
      node tmp244 = eq(UInt<12>(1310), addr)
      node tmp245 = mux(tmp244, reg_tohost238, tmp243)
      data := tmp245
      node tmp246 = eq(cmd, UInt<2>(1))
      when tmp246 :
         node tmp247 = eq(addr, UInt<12>(1310))
         when tmp247 :
            reg_tohost238 := src         
         node tmp248 = eq(addr, UInt<12>(1290))
         when tmp248 :
            reg_status239 := src      
      node tmp249 = eq(cmd, UInt<2>(2))
      node tmp250 = neq(src, UInt<1>(0))
      node tmp251 = bit-and(tmp249, tmp250)
      when tmp251 :
         node tmp252 = eq(addr, UInt<12>(1310))
         when tmp252 :
            node tmp253 = dshl(UInt<1>(1), src)
            node tmp254 = bit-or(data, tmp253)
            reg_tohost238 := tmp254         
         node tmp255 = eq(addr, UInt<12>(1290))
         when tmp255 :
            node tmp256 = dshl(UInt<1>(1), src)
            node tmp257 = bit-or(data, tmp256)
            reg_status239 := tmp257      
      node tmp258 = eq(cmd, UInt<2>(3))
      node tmp259 = neq(src, UInt<1>(0))
      node tmp260 = bit-and(tmp258, tmp259)
      when tmp260 :
         node tmp261 = eq(addr, UInt<12>(1310))
         when tmp261 :
            node tmp262 = dshl(UInt<1>(0), src)
            node tmp263 = bit-and(data, tmp262)
            reg_tohost238 := tmp263         
         node tmp264 = eq(addr, UInt<12>(1290))
         when tmp264 :
            node tmp265 = dshl(UInt<1>(0), src)
            node tmp266 = bit-and(data, tmp265)
            reg_status239 := tmp266   
   module Datapath :
      input stall : UInt<1>
      input ctrl : {pc_sel : UInt<1>, inst_re : UInt<1>, inst_type : UInt<1>, A_sel : UInt<1>, B_sel : UInt<1>, imm_sel : UInt<3>, alu_op : UInt<4>, br_type : UInt<3>, data_re : UInt<1>, st_type : UInt<2>, ld_type : UInt<3>, wb_sel : UInt<2>, wb_en : UInt<1>, csr_cmd : UInt<2>, flip ins : UInt<32>, flip stall : UInt<1>}
      output host : {flip hid : UInt<1>, tohost : UInt<32>, status : UInt<32>}
      output icache : {addr : UInt<32>, din : UInt<32>, flip dout : UInt<32>, re : UInt<1>, we : UInt<4>}
      output dcache : {addr : UInt<32>, din : UInt<32>, flip dout : UInt<32>, re : UInt<1>, we : UInt<4>}
      inst alu of ALU
      inst brCond of BrCond
      inst regFile of RegFile
      inst immGen of ImmGenWire
      reg fe_inst137 : UInt<32>
      on-reset fe_inst137 := UInt<1>(0)
      reg fe_pc138 : UInt<?>
      reg ew_inst139 : UInt<32>
      on-reset ew_inst139 := UInt<1>(0)
      reg ew_pc140 : UInt<?>
      reg ew_alu141 : UInt<?>
      reg pc142 : UInt<32>
      node tmp143 = sub-wrap(UInt<14>(8192), UInt<32>(4))
      on-reset pc142 := tmp143
      node tmp144 = eq(ctrl.pc_sel, UInt<1>(1))
      node tmp145 = bit-or(tmp144, brCond.taken)
      node tmp146 = add-wrap(pc142, UInt<3>(4))
      node tmp147 = mux(tmp145, alu.sum, tmp146)
      node tmp148 = eq(ctrl.inst_type, UInt<1>(1))
      node tmp149 = bit-or(tmp148, brCond.taken)
      node tmp150 = mux(tmp149, UInt<5>(19), icache.dout)
      icache.we := UInt<1>(0)
      icache.din := UInt<1>(0)
      icache.addr := tmp147
      icache.re := ctrl.inst_re
      node tmp151 = eq(dcache.we, UInt<1>(0))
      node tmp152 = bit-not(tmp151)
      node tmp153 = bit-not(tmp152)
      node tmp154 = bit-and(icache.re, tmp153)
      node tmp155 = mux(tmp154, tmp147, pc142)
      pc142 := tmp155
      node tmp156 = bit-not(stall)
      when tmp156 :
         fe_pc138 := pc142
         fe_inst137 := tmp150      
      ctrl.ins := fe_inst137
      ctrl.stall := stall
      node tmp157 = bits(fe_inst137, 11, 7)
      node rd_addr158 = tmp157
      node tmp159 = bits(fe_inst137, 19, 15)
      node rs1_addr160 = tmp159
      node tmp161 = bits(fe_inst137, 24, 20)
      node rs2_addr162 = tmp161
      regFile.raddr1 := rs1_addr160
      regFile.raddr2 := rs2_addr162
      immGen.inst := fe_inst137
      immGen.sel := ctrl.imm_sel
      node tmp163 = eq(rs1_addr160, UInt<1>(0))
      node tmp164 = bit-not(tmp163)
      node rs1NotZero165 = tmp164
      node tmp166 = eq(rs2_addr162, UInt<1>(0))
      node tmp167 = bit-not(tmp166)
      node rs2NotZero168 = tmp167
      node tmp169 = eq(ctrl.wb_sel, UInt<2>(0))
      node tmp170 = bit-and(ctrl.wb_en, tmp169)
      node alutype171 = tmp170
      node tmp172 = bits(ew_inst139, 11, 7)
      node ex_rd_addr173 = tmp172
      node tmp174 = eq(rs1_addr160, ex_rd_addr173)
      node tmp175 = bit-and(rs1NotZero165, tmp174)
      node tmp176 = bit-and(alutype171, tmp175)
      node tmp177 = mux(tmp176, ew_alu141, regFile.rdata1)
      node rs1178 = tmp177
      node tmp179 = eq(rs2_addr162, ex_rd_addr173)
      node tmp180 = bit-and(rs2NotZero168, tmp179)
      node tmp181 = bit-and(alutype171, tmp180)
      node tmp182 = mux(tmp181, ew_alu141, regFile.rdata2)
      node rs2183 = tmp182
      node tmp184 = eq(ctrl.A_sel, UInt<1>(0))
      node tmp185 = mux(tmp184, rs1178, fe_pc138)
      alu.A := tmp185
      node tmp186 = eq(ctrl.B_sel, UInt<1>(0))
      node tmp187 = mux(tmp186, rs2183, immGen.out)
      alu.B := tmp187
      alu.alu_op := ctrl.alu_op
      brCond.rs1 := rs1178
      brCond.rs2 := rs2183
      brCond.br_type := ctrl.br_type
      node tmp188 = bit(alu.sum, 1)
      node tmp189 = shl(tmp188, 4)
      node tmp190 = bit(alu.sum, 0)
      node tmp191 = shl(tmp190, 3)
      node tmp192 = bit-or(tmp189, tmp191)
      node woffset193 = tmp192
      dcache.re := ctrl.data_re
      node tmp194 = mux(stall, ew_alu141, alu.sum)
      dcache.addr := tmp194
      node tmp195 = bits(alu.sum, 1, 0)
      node tmp196 = dshl(UInt<2>(3), tmp195)
      node tmp197 = bits(tmp196, 3, 0)
      node tmp198 = bits(alu.sum, 1, 0)
      node tmp199 = dshl(UInt<1>(1), tmp198)
      node tmp200 = bits(tmp199, 3, 0)
      node tmp201 = eq(UInt<2>(2), ctrl.st_type)
      node tmp202 = mux(tmp201, tmp200, UInt<1>(0))
      node tmp203 = eq(UInt<2>(1), ctrl.st_type)
      node tmp204 = mux(tmp203, tmp197, tmp202)
      node tmp205 = eq(UInt<2>(0), ctrl.st_type)
      node tmp206 = mux(tmp205, UInt<4>(15), tmp204)
      node tmp207 = mux(stall, UInt<1>(0), tmp206)
      dcache.we := tmp207
      node tmp208 = dshl(rs2183, woffset193)
      node tmp209 = bits(tmp208, 31, 0)
      dcache.din := tmp209
      node tmp210 = bit-not(stall)
      when tmp210 :
         ew_pc140 := fe_pc138
         ew_inst139 := fe_inst137
         ew_alu141 := alu.out      
      node tmp211 = bit(ew_alu141, 1)
      node tmp212 = shl(tmp211, 4)
      node tmp213 = bit(ew_alu141, 0)
      node tmp214 = shl(tmp213, 3)
      node tmp215 = bit-or(tmp212, tmp214)
      node loffset216 = tmp215
      node tmp217 = dshr(dcache.dout, loffset216)
      node lshift218 = tmp217
      node tmp219 = bits(lshift218, 15, 0)
      node tmp220 = convert(tmp219)
      node tmp221 = pad(tmp220, 32)
      node tmp222 = as-UInt(tmp221)
      node tmp223 = bits(lshift218, 7, 0)
      node tmp224 = convert(tmp223)
      node tmp225 = pad(tmp224, 32)
      node tmp226 = as-UInt(tmp225)
      node tmp227 = bits(lshift218, 15, 0)
      node tmp228 = bits(lshift218, 7, 0)
      node tmp229 = eq(UInt<3>(4), ctrl.ld_type)
      node tmp230 = mux(tmp229, tmp228, dcache.dout)
      node tmp231 = eq(UInt<3>(3), ctrl.ld_type)
      node tmp232 = mux(tmp231, tmp227, tmp230)
      node tmp233 = eq(UInt<3>(2), ctrl.ld_type)
      node tmp234 = mux(tmp233, tmp226, tmp232)
      node tmp235 = eq(UInt<3>(1), ctrl.ld_type)
      node tmp236 = mux(tmp235, tmp222, tmp234)
      node load237 = tmp236
      inst csr of CSR
      host := csr.host
      csr.src := ew_alu141
      node tmp267 = bits(ew_inst139, 31, 20)
      csr.addr := tmp267
      csr.cmd := ctrl.csr_cmd
      node tmp268 = add-wrap(ew_pc140, UInt<3>(4))
      node tmp269 = eq(UInt<2>(3), ctrl.wb_sel)
      node tmp270 = mux(tmp269, csr.data, ew_alu141)
      node tmp271 = eq(UInt<2>(2), ctrl.wb_sel)
      node tmp272 = mux(tmp271, tmp268, tmp270)
      node tmp273 = eq(UInt<2>(1), ctrl.wb_sel)
      node tmp274 = mux(tmp273, load237, tmp272)
      node regWrite275 = tmp274
      regFile.wen := ctrl.wb_en
      regFile.waddr := ex_rd_addr173
      regFile.wdata := regWrite275   
   module Control :
      output ctrl : {pc_sel : UInt<1>, inst_re : UInt<1>, inst_type : UInt<1>, A_sel : UInt<1>, B_sel : UInt<1>, imm_sel : UInt<3>, alu_op : UInt<4>, br_type : UInt<3>, data_re : UInt<1>, st_type : UInt<2>, ld_type : UInt<3>, wb_sel : UInt<2>, wb_en : UInt<1>, csr_cmd : UInt<2>, flip ins : UInt<32>, flip stall : UInt<1>}
      node tmp276 = bit-and(UInt<32>(127), ctrl.ins)
      node tmp277 = eq(tmp276, UInt<32>(55))
      node tmp278 = bit-and(UInt<32>(127), ctrl.ins)
      node tmp279 = eq(tmp278, UInt<32>(23))
      node tmp280 = bit-and(UInt<32>(127), ctrl.ins)
      node tmp281 = eq(tmp280, UInt<32>(111))
      node tmp282 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp283 = eq(tmp282, UInt<32>(103))
      node tmp284 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp285 = eq(tmp284, UInt<32>(99))
      node tmp286 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp287 = eq(tmp286, UInt<32>(4195))
      node tmp288 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp289 = eq(tmp288, UInt<32>(16483))
      node tmp290 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp291 = eq(tmp290, UInt<32>(20579))
      node tmp292 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp293 = eq(tmp292, UInt<32>(24675))
      node tmp294 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp295 = eq(tmp294, UInt<32>(28771))
      node tmp296 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp297 = eq(tmp296, UInt<32>(3))
      node tmp298 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp299 = eq(tmp298, UInt<32>(4099))
      node tmp300 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp301 = eq(tmp300, UInt<32>(8195))
      node tmp302 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp303 = eq(tmp302, UInt<32>(16387))
      node tmp304 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp305 = eq(tmp304, UInt<32>(20483))
      node tmp306 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp307 = eq(tmp306, UInt<32>(35))
      node tmp308 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp309 = eq(tmp308, UInt<32>(4131))
      node tmp310 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp311 = eq(tmp310, UInt<32>(8227))
      node tmp312 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp313 = eq(tmp312, UInt<32>(19))
      node tmp314 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp315 = eq(tmp314, UInt<32>(8211))
      node tmp316 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp317 = eq(tmp316, UInt<32>(12307))
      node tmp318 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp319 = eq(tmp318, UInt<32>(16403))
      node tmp320 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp321 = eq(tmp320, UInt<32>(24595))
      node tmp322 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp323 = eq(tmp322, UInt<32>(28691))
      node tmp324 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp325 = eq(tmp324, UInt<32>(4115))
      node tmp326 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp327 = eq(tmp326, UInt<32>(20499))
      node tmp328 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp329 = eq(tmp328, UInt<32>(1073762323))
      node tmp330 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp331 = eq(tmp330, UInt<32>(51))
      node tmp332 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp333 = eq(tmp332, UInt<32>(1073741875))
      node tmp334 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp335 = eq(tmp334, UInt<32>(4147))
      node tmp336 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp337 = eq(tmp336, UInt<32>(8243))
      node tmp338 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp339 = eq(tmp338, UInt<32>(12339))
      node tmp340 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp341 = eq(tmp340, UInt<32>(16435))
      node tmp342 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp343 = eq(tmp342, UInt<32>(20531))
      node tmp344 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp345 = eq(tmp344, UInt<32>(1073762355))
      node tmp346 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp347 = eq(tmp346, UInt<32>(24627))
      node tmp348 = bit-and(UInt<32>(-33525633), ctrl.ins)
      node tmp349 = eq(tmp348, UInt<32>(28723))
      node tmp350 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp351 = eq(tmp350, UInt<32>(4211))
      node tmp352 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp353 = eq(tmp352, UInt<32>(8307))
      node tmp354 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp355 = eq(tmp354, UInt<32>(12403))
      node tmp356 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp357 = eq(tmp356, UInt<32>(20595))
      node tmp358 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp359 = eq(tmp358, UInt<32>(24691))
      node tmp360 = bit-and(UInt<32>(28799), ctrl.ins)
      node tmp361 = eq(tmp360, UInt<32>(28787))
      node tmp362 = mux(tmp361, UInt<1>(0), UInt<1>(0))
      node tmp363 = mux(tmp359, UInt<1>(0), tmp362)
      node tmp364 = mux(tmp357, UInt<1>(0), tmp363)
      node tmp365 = mux(tmp355, UInt<1>(0), tmp364)
      node tmp366 = mux(tmp353, UInt<1>(0), tmp365)
      node tmp367 = mux(tmp351, UInt<1>(0), tmp366)
      node tmp368 = mux(tmp349, UInt<1>(0), tmp367)
      node tmp369 = mux(tmp347, UInt<1>(0), tmp368)
      node tmp370 = mux(tmp345, UInt<1>(0), tmp369)
      node tmp371 = mux(tmp343, UInt<1>(0), tmp370)
      node tmp372 = mux(tmp341, UInt<1>(0), tmp371)
      node tmp373 = mux(tmp339, UInt<1>(0), tmp372)
      node tmp374 = mux(tmp337, UInt<1>(0), tmp373)
      node tmp375 = mux(tmp335, UInt<1>(0), tmp374)
      node tmp376 = mux(tmp333, UInt<1>(0), tmp375)
      node tmp377 = mux(tmp331, UInt<1>(0), tmp376)
      node tmp378 = mux(tmp329, UInt<1>(0), tmp377)
      node tmp379 = mux(tmp327, UInt<1>(0), tmp378)
      node tmp380 = mux(tmp325, UInt<1>(0), tmp379)
      node tmp381 = mux(tmp323, UInt<1>(0), tmp380)
      node tmp382 = mux(tmp321, UInt<1>(0), tmp381)
      node tmp383 = mux(tmp319, UInt<1>(0), tmp382)
      node tmp384 = mux(tmp317, UInt<1>(0), tmp383)
      node tmp385 = mux(tmp315, UInt<1>(0), tmp384)
      node tmp386 = mux(tmp313, UInt<1>(0), tmp385)
      node tmp387 = mux(tmp311, UInt<1>(0), tmp386)
      node tmp388 = mux(tmp309, UInt<1>(0), tmp387)
      node tmp389 = mux(tmp307, UInt<1>(0), tmp388)
      node tmp390 = mux(tmp305, UInt<1>(0), tmp389)
      node tmp391 = mux(tmp303, UInt<1>(0), tmp390)
      node tmp392 = mux(tmp301, UInt<1>(0), tmp391)
      node tmp393 = mux(tmp299, UInt<1>(0), tmp392)
      node tmp394 = mux(tmp297, UInt<1>(0), tmp393)
      node tmp395 = mux(tmp295, UInt<1>(0), tmp394)
      node tmp396 = mux(tmp293, UInt<1>(0), tmp395)
      node tmp397 = mux(tmp291, UInt<1>(0), tmp396)
      node tmp398 = mux(tmp289, UInt<1>(0), tmp397)
      node tmp399 = mux(tmp287, UInt<1>(0), tmp398)
      node tmp400 = mux(tmp285, UInt<1>(0), tmp399)
      node tmp401 = mux(tmp283, UInt<1>(1), tmp400)
      node tmp402 = mux(tmp281, UInt<1>(1), tmp401)
      node tmp403 = mux(tmp279, UInt<1>(0), tmp402)
      node tmp404 = mux(tmp277, UInt<1>(0), tmp403)
      node tmp405 = mux(tmp361, UInt<1>(1), UInt<1>(1))
      node tmp406 = mux(tmp359, UInt<1>(1), tmp405)
      node tmp407 = mux(tmp357, UInt<1>(1), tmp406)
      node tmp408 = mux(tmp355, UInt<1>(0), tmp407)
      node tmp409 = mux(tmp353, UInt<1>(0), tmp408)
      node tmp410 = mux(tmp351, UInt<1>(0), tmp409)
      node tmp411 = mux(tmp349, UInt<1>(0), tmp410)
      node tmp412 = mux(tmp347, UInt<1>(0), tmp411)
      node tmp413 = mux(tmp345, UInt<1>(0), tmp412)
      node tmp414 = mux(tmp343, UInt<1>(0), tmp413)
      node tmp415 = mux(tmp341, UInt<1>(0), tmp414)
      node tmp416 = mux(tmp339, UInt<1>(0), tmp415)
      node tmp417 = mux(tmp337, UInt<1>(0), tmp416)
      node tmp418 = mux(tmp335, UInt<1>(0), tmp417)
      node tmp419 = mux(tmp333, UInt<1>(0), tmp418)
      node tmp420 = mux(tmp331, UInt<1>(0), tmp419)
      node tmp421 = mux(tmp329, UInt<1>(0), tmp420)
      node tmp422 = mux(tmp327, UInt<1>(0), tmp421)
      node tmp423 = mux(tmp325, UInt<1>(0), tmp422)
      node tmp424 = mux(tmp323, UInt<1>(0), tmp423)
      node tmp425 = mux(tmp321, UInt<1>(0), tmp424)
      node tmp426 = mux(tmp319, UInt<1>(0), tmp425)
      node tmp427 = mux(tmp317, UInt<1>(0), tmp426)
      node tmp428 = mux(tmp315, UInt<1>(0), tmp427)
      node tmp429 = mux(tmp313, UInt<1>(0), tmp428)
      node tmp430 = mux(tmp311, UInt<1>(0), tmp429)
      node tmp431 = mux(tmp309, UInt<1>(0), tmp430)
      node tmp432 = mux(tmp307, UInt<1>(0), tmp431)
      node tmp433 = mux(tmp305, UInt<1>(0), tmp432)
      node tmp434 = mux(tmp303, UInt<1>(0), tmp433)
      node tmp435 = mux(tmp301, UInt<1>(0), tmp434)
      node tmp436 = mux(tmp299, UInt<1>(0), tmp435)
      node tmp437 = mux(tmp297, UInt<1>(0), tmp436)
      node tmp438 = mux(tmp295, UInt<1>(1), tmp437)
      node tmp439 = mux(tmp293, UInt<1>(1), tmp438)
      node tmp440 = mux(tmp291, UInt<1>(1), tmp439)
      node tmp441 = mux(tmp289, UInt<1>(1), tmp440)
      node tmp442 = mux(tmp287, UInt<1>(1), tmp441)
      node tmp443 = mux(tmp285, UInt<1>(1), tmp442)
      node tmp444 = mux(tmp283, UInt<1>(0), tmp443)
      node tmp445 = mux(tmp281, UInt<1>(1), tmp444)
      node tmp446 = mux(tmp279, UInt<1>(1), tmp445)
      node tmp447 = mux(tmp277, UInt<1>(1), tmp446)
      node tmp448 = mux(tmp361, UInt<1>(1), UInt<1>(0))
      node tmp449 = mux(tmp359, UInt<1>(1), tmp448)
      node tmp450 = mux(tmp357, UInt<1>(1), tmp449)
      node tmp451 = mux(tmp355, UInt<1>(0), tmp450)
      node tmp452 = mux(tmp353, UInt<1>(0), tmp451)
      node tmp453 = mux(tmp351, UInt<1>(0), tmp452)
      node tmp454 = mux(tmp349, UInt<1>(0), tmp453)
      node tmp455 = mux(tmp347, UInt<1>(0), tmp454)
      node tmp456 = mux(tmp345, UInt<1>(0), tmp455)
      node tmp457 = mux(tmp343, UInt<1>(0), tmp456)
      node tmp458 = mux(tmp341, UInt<1>(0), tmp457)
      node tmp459 = mux(tmp339, UInt<1>(0), tmp458)
      node tmp460 = mux(tmp337, UInt<1>(0), tmp459)
      node tmp461 = mux(tmp335, UInt<1>(0), tmp460)
      node tmp462 = mux(tmp333, UInt<1>(0), tmp461)
      node tmp463 = mux(tmp331, UInt<1>(0), tmp462)
      node tmp464 = mux(tmp329, UInt<1>(1), tmp463)
      node tmp465 = mux(tmp327, UInt<1>(1), tmp464)
      node tmp466 = mux(tmp325, UInt<1>(1), tmp465)
      node tmp467 = mux(tmp323, UInt<1>(1), tmp466)
      node tmp468 = mux(tmp321, UInt<1>(1), tmp467)
      node tmp469 = mux(tmp319, UInt<1>(1), tmp468)
      node tmp470 = mux(tmp317, UInt<1>(1), tmp469)
      node tmp471 = mux(tmp315, UInt<1>(1), tmp470)
      node tmp472 = mux(tmp313, UInt<1>(1), tmp471)
      node tmp473 = mux(tmp311, UInt<1>(1), tmp472)
      node tmp474 = mux(tmp309, UInt<1>(1), tmp473)
      node tmp475 = mux(tmp307, UInt<1>(1), tmp474)
      node tmp476 = mux(tmp305, UInt<1>(1), tmp475)
      node tmp477 = mux(tmp303, UInt<1>(1), tmp476)
      node tmp478 = mux(tmp301, UInt<1>(1), tmp477)
      node tmp479 = mux(tmp299, UInt<1>(1), tmp478)
      node tmp480 = mux(tmp297, UInt<1>(1), tmp479)
      node tmp481 = mux(tmp295, UInt<1>(1), tmp480)
      node tmp482 = mux(tmp293, UInt<1>(1), tmp481)
      node tmp483 = mux(tmp291, UInt<1>(1), tmp482)
      node tmp484 = mux(tmp289, UInt<1>(1), tmp483)
      node tmp485 = mux(tmp287, UInt<1>(1), tmp484)
      node tmp486 = mux(tmp285, UInt<1>(1), tmp485)
      node tmp487 = mux(tmp283, UInt<1>(1), tmp486)
      node tmp488 = mux(tmp281, UInt<1>(1), tmp487)
      node tmp489 = mux(tmp279, UInt<1>(1), tmp488)
      node tmp490 = mux(tmp277, UInt<1>(1), tmp489)
      node tmp491 = mux(tmp361, UInt<3>(5), UInt<3>(7))
      node tmp492 = mux(tmp359, UInt<3>(5), tmp491)
      node tmp493 = mux(tmp357, UInt<3>(5), tmp492)
      node tmp494 = mux(tmp355, UInt<3>(5), tmp493)
      node tmp495 = mux(tmp353, UInt<3>(5), tmp494)
      node tmp496 = mux(tmp351, UInt<3>(5), tmp495)
      node tmp497 = mux(tmp349, UInt<3>(7), tmp496)
      node tmp498 = mux(tmp347, UInt<3>(7), tmp497)
      node tmp499 = mux(tmp345, UInt<3>(7), tmp498)
      node tmp500 = mux(tmp343, UInt<3>(7), tmp499)
      node tmp501 = mux(tmp341, UInt<3>(7), tmp500)
      node tmp502 = mux(tmp339, UInt<3>(7), tmp501)
      node tmp503 = mux(tmp337, UInt<3>(7), tmp502)
      node tmp504 = mux(tmp335, UInt<3>(7), tmp503)
      node tmp505 = mux(tmp333, UInt<3>(7), tmp504)
      node tmp506 = mux(tmp331, UInt<3>(7), tmp505)
      node tmp507 = mux(tmp329, UInt<3>(0), tmp506)
      node tmp508 = mux(tmp327, UInt<3>(0), tmp507)
      node tmp509 = mux(tmp325, UInt<3>(0), tmp508)
      node tmp510 = mux(tmp323, UInt<3>(0), tmp509)
      node tmp511 = mux(tmp321, UInt<3>(0), tmp510)
      node tmp512 = mux(tmp319, UInt<3>(0), tmp511)
      node tmp513 = mux(tmp317, UInt<3>(0), tmp512)
      node tmp514 = mux(tmp315, UInt<3>(0), tmp513)
      node tmp515 = mux(tmp313, UInt<3>(0), tmp514)
      node tmp516 = mux(tmp311, UInt<3>(1), tmp515)
      node tmp517 = mux(tmp309, UInt<3>(1), tmp516)
      node tmp518 = mux(tmp307, UInt<3>(1), tmp517)
      node tmp519 = mux(tmp305, UInt<3>(0), tmp518)
      node tmp520 = mux(tmp303, UInt<3>(0), tmp519)
      node tmp521 = mux(tmp301, UInt<3>(0), tmp520)
      node tmp522 = mux(tmp299, UInt<3>(0), tmp521)
      node tmp523 = mux(tmp297, UInt<3>(0), tmp522)
      node tmp524 = mux(tmp295, UInt<3>(4), tmp523)
      node tmp525 = mux(tmp293, UInt<3>(4), tmp524)
      node tmp526 = mux(tmp291, UInt<3>(4), tmp525)
      node tmp527 = mux(tmp289, UInt<3>(4), tmp526)
      node tmp528 = mux(tmp287, UInt<3>(4), tmp527)
      node tmp529 = mux(tmp285, UInt<3>(4), tmp528)
      node tmp530 = mux(tmp283, UInt<3>(0), tmp529)
      node tmp531 = mux(tmp281, UInt<3>(3), tmp530)
      node tmp532 = mux(tmp279, UInt<3>(2), tmp531)
      node tmp533 = mux(tmp277, UInt<3>(2), tmp532)
      node tmp534 = mux(tmp361, UInt<4>(11), UInt<4>(15))
      node tmp535 = mux(tmp359, UInt<4>(11), tmp534)
      node tmp536 = mux(tmp357, UInt<4>(11), tmp535)
      node tmp537 = mux(tmp355, UInt<4>(10), tmp536)
      node tmp538 = mux(tmp353, UInt<4>(10), tmp537)
      node tmp539 = mux(tmp351, UInt<4>(10), tmp538)
      node tmp540 = mux(tmp349, UInt<4>(2), tmp539)
      node tmp541 = mux(tmp347, UInt<4>(3), tmp540)
      node tmp542 = mux(tmp345, UInt<4>(9), tmp541)
      node tmp543 = mux(tmp343, UInt<4>(8), tmp542)
      node tmp544 = mux(tmp341, UInt<4>(4), tmp543)
      node tmp545 = mux(tmp339, UInt<4>(7), tmp544)
      node tmp546 = mux(tmp337, UInt<4>(5), tmp545)
      node tmp547 = mux(tmp335, UInt<4>(6), tmp546)
      node tmp548 = mux(tmp333, UInt<4>(1), tmp547)
      node tmp549 = mux(tmp331, UInt<4>(0), tmp548)
      node tmp550 = mux(tmp329, UInt<4>(9), tmp549)
      node tmp551 = mux(tmp327, UInt<4>(8), tmp550)
      node tmp552 = mux(tmp325, UInt<4>(6), tmp551)
      node tmp553 = mux(tmp323, UInt<4>(2), tmp552)
      node tmp554 = mux(tmp321, UInt<4>(3), tmp553)
      node tmp555 = mux(tmp319, UInt<4>(4), tmp554)
      node tmp556 = mux(tmp317, UInt<4>(7), tmp555)
      node tmp557 = mux(tmp315, UInt<4>(5), tmp556)
      node tmp558 = mux(tmp313, UInt<4>(0), tmp557)
      node tmp559 = mux(tmp311, UInt<4>(0), tmp558)
      node tmp560 = mux(tmp309, UInt<4>(0), tmp559)
      node tmp561 = mux(tmp307, UInt<4>(0), tmp560)
      node tmp562 = mux(tmp305, UInt<4>(0), tmp561)
      node tmp563 = mux(tmp303, UInt<4>(0), tmp562)
      node tmp564 = mux(tmp301, UInt<4>(0), tmp563)
      node tmp565 = mux(tmp299, UInt<4>(0), tmp564)
      node tmp566 = mux(tmp297, UInt<4>(0), tmp565)
      node tmp567 = mux(tmp295, UInt<4>(0), tmp566)
      node tmp568 = mux(tmp293, UInt<4>(0), tmp567)
      node tmp569 = mux(tmp291, UInt<4>(0), tmp568)
      node tmp570 = mux(tmp289, UInt<4>(0), tmp569)
      node tmp571 = mux(tmp287, UInt<4>(0), tmp570)
      node tmp572 = mux(tmp285, UInt<4>(0), tmp571)
      node tmp573 = mux(tmp283, UInt<4>(0), tmp572)
      node tmp574 = mux(tmp281, UInt<4>(0), tmp573)
      node tmp575 = mux(tmp279, UInt<4>(0), tmp574)
      node tmp576 = mux(tmp277, UInt<4>(11), tmp575)
      node tmp577 = mux(tmp361, UInt<3>(7), UInt<3>(7))
      node tmp578 = mux(tmp359, UInt<3>(7), tmp577)
      node tmp579 = mux(tmp357, UInt<3>(7), tmp578)
      node tmp580 = mux(tmp355, UInt<3>(7), tmp579)
      node tmp581 = mux(tmp353, UInt<3>(7), tmp580)
      node tmp582 = mux(tmp351, UInt<3>(7), tmp581)
      node tmp583 = mux(tmp349, UInt<3>(7), tmp582)
      node tmp584 = mux(tmp347, UInt<3>(7), tmp583)
      node tmp585 = mux(tmp345, UInt<3>(7), tmp584)
      node tmp586 = mux(tmp343, UInt<3>(7), tmp585)
      node tmp587 = mux(tmp341, UInt<3>(7), tmp586)
      node tmp588 = mux(tmp339, UInt<3>(7), tmp587)
      node tmp589 = mux(tmp337, UInt<3>(7), tmp588)
      node tmp590 = mux(tmp335, UInt<3>(7), tmp589)
      node tmp591 = mux(tmp333, UInt<3>(7), tmp590)
      node tmp592 = mux(tmp331, UInt<3>(7), tmp591)
      node tmp593 = mux(tmp329, UInt<3>(7), tmp592)
      node tmp594 = mux(tmp327, UInt<3>(7), tmp593)
      node tmp595 = mux(tmp325, UInt<3>(7), tmp594)
      node tmp596 = mux(tmp323, UInt<3>(7), tmp595)
      node tmp597 = mux(tmp321, UInt<3>(7), tmp596)
      node tmp598 = mux(tmp319, UInt<3>(7), tmp597)
      node tmp599 = mux(tmp317, UInt<3>(7), tmp598)
      node tmp600 = mux(tmp315, UInt<3>(7), tmp599)
      node tmp601 = mux(tmp313, UInt<3>(7), tmp600)
      node tmp602 = mux(tmp311, UInt<3>(7), tmp601)
      node tmp603 = mux(tmp309, UInt<3>(7), tmp602)
      node tmp604 = mux(tmp307, UInt<3>(7), tmp603)
      node tmp605 = mux(tmp305, UInt<3>(7), tmp604)
      node tmp606 = mux(tmp303, UInt<3>(7), tmp605)
      node tmp607 = mux(tmp301, UInt<3>(7), tmp606)
      node tmp608 = mux(tmp299, UInt<3>(7), tmp607)
      node tmp609 = mux(tmp297, UInt<3>(7), tmp608)
      node tmp610 = mux(tmp295, UInt<3>(4), tmp609)
      node tmp611 = mux(tmp293, UInt<3>(0), tmp610)
      node tmp612 = mux(tmp291, UInt<3>(5), tmp611)
      node tmp613 = mux(tmp289, UInt<3>(1), tmp612)
      node tmp614 = mux(tmp287, UInt<3>(6), tmp613)
      node tmp615 = mux(tmp285, UInt<3>(2), tmp614)
      node tmp616 = mux(tmp283, UInt<3>(7), tmp615)
      node tmp617 = mux(tmp281, UInt<3>(7), tmp616)
      node tmp618 = mux(tmp279, UInt<3>(7), tmp617)
      node tmp619 = mux(tmp277, UInt<3>(7), tmp618)
      node tmp620 = mux(tmp361, UInt<1>(0), UInt<1>(0))
      node tmp621 = mux(tmp359, UInt<1>(0), tmp620)
      node tmp622 = mux(tmp357, UInt<1>(0), tmp621)
      node tmp623 = mux(tmp355, UInt<1>(0), tmp622)
      node tmp624 = mux(tmp353, UInt<1>(0), tmp623)
      node tmp625 = mux(tmp351, UInt<1>(0), tmp624)
      node tmp626 = mux(tmp349, UInt<1>(0), tmp625)
      node tmp627 = mux(tmp347, UInt<1>(0), tmp626)
      node tmp628 = mux(tmp345, UInt<1>(0), tmp627)
      node tmp629 = mux(tmp343, UInt<1>(0), tmp628)
      node tmp630 = mux(tmp341, UInt<1>(0), tmp629)
      node tmp631 = mux(tmp339, UInt<1>(0), tmp630)
      node tmp632 = mux(tmp337, UInt<1>(0), tmp631)
      node tmp633 = mux(tmp335, UInt<1>(0), tmp632)
      node tmp634 = mux(tmp333, UInt<1>(0), tmp633)
      node tmp635 = mux(tmp331, UInt<1>(0), tmp634)
      node tmp636 = mux(tmp329, UInt<1>(0), tmp635)
      node tmp637 = mux(tmp327, UInt<1>(0), tmp636)
      node tmp638 = mux(tmp325, UInt<1>(0), tmp637)
      node tmp639 = mux(tmp323, UInt<1>(0), tmp638)
      node tmp640 = mux(tmp321, UInt<1>(0), tmp639)
      node tmp641 = mux(tmp319, UInt<1>(0), tmp640)
      node tmp642 = mux(tmp317, UInt<1>(0), tmp641)
      node tmp643 = mux(tmp315, UInt<1>(0), tmp642)
      node tmp644 = mux(tmp313, UInt<1>(0), tmp643)
      node tmp645 = mux(tmp311, UInt<1>(0), tmp644)
      node tmp646 = mux(tmp309, UInt<1>(0), tmp645)
      node tmp647 = mux(tmp307, UInt<1>(0), tmp646)
      node tmp648 = mux(tmp305, UInt<1>(0), tmp647)
      node tmp649 = mux(tmp303, UInt<1>(0), tmp648)
      node tmp650 = mux(tmp301, UInt<1>(0), tmp649)
      node tmp651 = mux(tmp299, UInt<1>(0), tmp650)
      node tmp652 = mux(tmp297, UInt<1>(0), tmp651)
      node tmp653 = mux(tmp295, UInt<1>(0), tmp652)
      node tmp654 = mux(tmp293, UInt<1>(0), tmp653)
      node tmp655 = mux(tmp291, UInt<1>(0), tmp654)
      node tmp656 = mux(tmp289, UInt<1>(0), tmp655)
      node tmp657 = mux(tmp287, UInt<1>(0), tmp656)
      node tmp658 = mux(tmp285, UInt<1>(0), tmp657)
      node tmp659 = mux(tmp283, UInt<1>(1), tmp658)
      node tmp660 = mux(tmp281, UInt<1>(1), tmp659)
      node tmp661 = mux(tmp279, UInt<1>(0), tmp660)
      node tmp662 = mux(tmp277, UInt<1>(0), tmp661)
      node tmp663 = mux(tmp361, UInt<2>(3), UInt<2>(3))
      node tmp664 = mux(tmp359, UInt<2>(3), tmp663)
      node tmp665 = mux(tmp357, UInt<2>(3), tmp664)
      node tmp666 = mux(tmp355, UInt<2>(3), tmp665)
      node tmp667 = mux(tmp353, UInt<2>(3), tmp666)
      node tmp668 = mux(tmp351, UInt<2>(3), tmp667)
      node tmp669 = mux(tmp349, UInt<2>(3), tmp668)
      node tmp670 = mux(tmp347, UInt<2>(3), tmp669)
      node tmp671 = mux(tmp345, UInt<2>(3), tmp670)
      node tmp672 = mux(tmp343, UInt<2>(3), tmp671)
      node tmp673 = mux(tmp341, UInt<2>(3), tmp672)
      node tmp674 = mux(tmp339, UInt<2>(3), tmp673)
      node tmp675 = mux(tmp337, UInt<2>(3), tmp674)
      node tmp676 = mux(tmp335, UInt<2>(3), tmp675)
      node tmp677 = mux(tmp333, UInt<2>(3), tmp676)
      node tmp678 = mux(tmp331, UInt<2>(3), tmp677)
      node tmp679 = mux(tmp329, UInt<2>(3), tmp678)
      node tmp680 = mux(tmp327, UInt<2>(3), tmp679)
      node tmp681 = mux(tmp325, UInt<2>(3), tmp680)
      node tmp682 = mux(tmp323, UInt<2>(3), tmp681)
      node tmp683 = mux(tmp321, UInt<2>(3), tmp682)
      node tmp684 = mux(tmp319, UInt<2>(3), tmp683)
      node tmp685 = mux(tmp317, UInt<2>(3), tmp684)
      node tmp686 = mux(tmp315, UInt<2>(3), tmp685)
      node tmp687 = mux(tmp313, UInt<2>(3), tmp686)
      node tmp688 = mux(tmp311, UInt<2>(0), tmp687)
      node tmp689 = mux(tmp309, UInt<2>(1), tmp688)
      node tmp690 = mux(tmp307, UInt<2>(2), tmp689)
      node tmp691 = mux(tmp305, UInt<2>(3), tmp690)
      node tmp692 = mux(tmp303, UInt<2>(3), tmp691)
      node tmp693 = mux(tmp301, UInt<2>(3), tmp692)
      node tmp694 = mux(tmp299, UInt<2>(3), tmp693)
      node tmp695 = mux(tmp297, UInt<2>(3), tmp694)
      node tmp696 = mux(tmp295, UInt<2>(3), tmp695)
      node tmp697 = mux(tmp293, UInt<2>(3), tmp696)
      node tmp698 = mux(tmp291, UInt<2>(3), tmp697)
      node tmp699 = mux(tmp289, UInt<2>(3), tmp698)
      node tmp700 = mux(tmp287, UInt<2>(3), tmp699)
      node tmp701 = mux(tmp285, UInt<2>(3), tmp700)
      node tmp702 = mux(tmp283, UInt<2>(3), tmp701)
      node tmp703 = mux(tmp281, UInt<2>(3), tmp702)
      node tmp704 = mux(tmp279, UInt<2>(3), tmp703)
      node tmp705 = mux(tmp277, UInt<2>(3), tmp704)
      node tmp706 = mux(tmp361, UInt<3>(7), UInt<3>(7))
      node tmp707 = mux(tmp359, UInt<3>(7), tmp706)
      node tmp708 = mux(tmp357, UInt<3>(7), tmp707)
      node tmp709 = mux(tmp355, UInt<3>(7), tmp708)
      node tmp710 = mux(tmp353, UInt<3>(7), tmp709)
      node tmp711 = mux(tmp351, UInt<3>(7), tmp710)
      node tmp712 = mux(tmp349, UInt<3>(7), tmp711)
      node tmp713 = mux(tmp347, UInt<3>(7), tmp712)
      node tmp714 = mux(tmp345, UInt<3>(7), tmp713)
      node tmp715 = mux(tmp343, UInt<3>(7), tmp714)
      node tmp716 = mux(tmp341, UInt<3>(7), tmp715)
      node tmp717 = mux(tmp339, UInt<3>(7), tmp716)
      node tmp718 = mux(tmp337, UInt<3>(7), tmp717)
      node tmp719 = mux(tmp335, UInt<3>(7), tmp718)
      node tmp720 = mux(tmp333, UInt<3>(7), tmp719)
      node tmp721 = mux(tmp331, UInt<3>(7), tmp720)
      node tmp722 = mux(tmp329, UInt<3>(7), tmp721)
      node tmp723 = mux(tmp327, UInt<3>(7), tmp722)
      node tmp724 = mux(tmp325, UInt<3>(7), tmp723)
      node tmp725 = mux(tmp323, UInt<3>(7), tmp724)
      node tmp726 = mux(tmp321, UInt<3>(7), tmp725)
      node tmp727 = mux(tmp319, UInt<3>(7), tmp726)
      node tmp728 = mux(tmp317, UInt<3>(7), tmp727)
      node tmp729 = mux(tmp315, UInt<3>(7), tmp728)
      node tmp730 = mux(tmp313, UInt<3>(7), tmp729)
      node tmp731 = mux(tmp311, UInt<3>(7), tmp730)
      node tmp732 = mux(tmp309, UInt<3>(7), tmp731)
      node tmp733 = mux(tmp307, UInt<3>(7), tmp732)
      node tmp734 = mux(tmp305, UInt<3>(3), tmp733)
      node tmp735 = mux(tmp303, UInt<3>(4), tmp734)
      node tmp736 = mux(tmp301, UInt<3>(0), tmp735)
      node tmp737 = mux(tmp299, UInt<3>(1), tmp736)
      node tmp738 = mux(tmp297, UInt<3>(2), tmp737)
      node tmp739 = mux(tmp295, UInt<3>(7), tmp738)
      node tmp740 = mux(tmp293, UInt<3>(7), tmp739)
      node tmp741 = mux(tmp291, UInt<3>(7), tmp740)
      node tmp742 = mux(tmp289, UInt<3>(7), tmp741)
      node tmp743 = mux(tmp287, UInt<3>(7), tmp742)
      node tmp744 = mux(tmp285, UInt<3>(7), tmp743)
      node tmp745 = mux(tmp283, UInt<3>(7), tmp744)
      node tmp746 = mux(tmp281, UInt<3>(7), tmp745)
      node tmp747 = mux(tmp279, UInt<3>(7), tmp746)
      node tmp748 = mux(tmp277, UInt<3>(7), tmp747)
      node tmp749 = mux(tmp361, UInt<2>(3), UInt<2>(0))
      node tmp750 = mux(tmp359, UInt<2>(3), tmp749)
      node tmp751 = mux(tmp357, UInt<2>(3), tmp750)
      node tmp752 = mux(tmp355, UInt<2>(3), tmp751)
      node tmp753 = mux(tmp353, UInt<2>(3), tmp752)
      node tmp754 = mux(tmp351, UInt<2>(3), tmp753)
      node tmp755 = mux(tmp349, UInt<2>(0), tmp754)
      node tmp756 = mux(tmp347, UInt<2>(0), tmp755)
      node tmp757 = mux(tmp345, UInt<2>(0), tmp756)
      node tmp758 = mux(tmp343, UInt<2>(0), tmp757)
      node tmp759 = mux(tmp341, UInt<2>(0), tmp758)
      node tmp760 = mux(tmp339, UInt<2>(0), tmp759)
      node tmp761 = mux(tmp337, UInt<2>(0), tmp760)
      node tmp762 = mux(tmp335, UInt<2>(0), tmp761)
      node tmp763 = mux(tmp333, UInt<2>(0), tmp762)
      node tmp764 = mux(tmp331, UInt<2>(0), tmp763)
      node tmp765 = mux(tmp329, UInt<2>(0), tmp764)
      node tmp766 = mux(tmp327, UInt<2>(0), tmp765)
      node tmp767 = mux(tmp325, UInt<2>(0), tmp766)
      node tmp768 = mux(tmp323, UInt<2>(0), tmp767)
      node tmp769 = mux(tmp321, UInt<2>(0), tmp768)
      node tmp770 = mux(tmp319, UInt<2>(0), tmp769)
      node tmp771 = mux(tmp317, UInt<2>(0), tmp770)
      node tmp772 = mux(tmp315, UInt<2>(0), tmp771)
      node tmp773 = mux(tmp313, UInt<2>(0), tmp772)
      node tmp774 = mux(tmp311, UInt<2>(0), tmp773)
      node tmp775 = mux(tmp309, UInt<2>(0), tmp774)
      node tmp776 = mux(tmp307, UInt<2>(0), tmp775)
      node tmp777 = mux(tmp305, UInt<2>(1), tmp776)
      node tmp778 = mux(tmp303, UInt<2>(1), tmp777)
      node tmp779 = mux(tmp301, UInt<2>(1), tmp778)
      node tmp780 = mux(tmp299, UInt<2>(1), tmp779)
      node tmp781 = mux(tmp297, UInt<2>(1), tmp780)
      node tmp782 = mux(tmp295, UInt<2>(0), tmp781)
      node tmp783 = mux(tmp293, UInt<2>(0), tmp782)
      node tmp784 = mux(tmp291, UInt<2>(0), tmp783)
      node tmp785 = mux(tmp289, UInt<2>(0), tmp784)
      node tmp786 = mux(tmp287, UInt<2>(0), tmp785)
      node tmp787 = mux(tmp285, UInt<2>(0), tmp786)
      node tmp788 = mux(tmp283, UInt<2>(2), tmp787)
      node tmp789 = mux(tmp281, UInt<2>(2), tmp788)
      node tmp790 = mux(tmp279, UInt<2>(0), tmp789)
      node tmp791 = mux(tmp277, UInt<2>(0), tmp790)
      node tmp792 = mux(tmp361, UInt<1>(0), UInt<1>(0))
      node tmp793 = mux(tmp359, UInt<1>(0), tmp792)
      node tmp794 = mux(tmp357, UInt<1>(0), tmp793)
      node tmp795 = mux(tmp355, UInt<1>(0), tmp794)
      node tmp796 = mux(tmp353, UInt<1>(0), tmp795)
      node tmp797 = mux(tmp351, UInt<1>(0), tmp796)
      node tmp798 = mux(tmp349, UInt<1>(1), tmp797)
      node tmp799 = mux(tmp347, UInt<1>(1), tmp798)
      node tmp800 = mux(tmp345, UInt<1>(1), tmp799)
      node tmp801 = mux(tmp343, UInt<1>(1), tmp800)
      node tmp802 = mux(tmp341, UInt<1>(1), tmp801)
      node tmp803 = mux(tmp339, UInt<1>(1), tmp802)
      node tmp804 = mux(tmp337, UInt<1>(1), tmp803)
      node tmp805 = mux(tmp335, UInt<1>(1), tmp804)
      node tmp806 = mux(tmp333, UInt<1>(1), tmp805)
      node tmp807 = mux(tmp331, UInt<1>(1), tmp806)
      node tmp808 = mux(tmp329, UInt<1>(1), tmp807)
      node tmp809 = mux(tmp327, UInt<1>(1), tmp808)
      node tmp810 = mux(tmp325, UInt<1>(1), tmp809)
      node tmp811 = mux(tmp323, UInt<1>(1), tmp810)
      node tmp812 = mux(tmp321, UInt<1>(1), tmp811)
      node tmp813 = mux(tmp319, UInt<1>(1), tmp812)
      node tmp814 = mux(tmp317, UInt<1>(1), tmp813)
      node tmp815 = mux(tmp315, UInt<1>(1), tmp814)
      node tmp816 = mux(tmp313, UInt<1>(1), tmp815)
      node tmp817 = mux(tmp311, UInt<1>(0), tmp816)
      node tmp818 = mux(tmp309, UInt<1>(0), tmp817)
      node tmp819 = mux(tmp307, UInt<1>(0), tmp818)
      node tmp820 = mux(tmp305, UInt<1>(1), tmp819)
      node tmp821 = mux(tmp303, UInt<1>(1), tmp820)
      node tmp822 = mux(tmp301, UInt<1>(1), tmp821)
      node tmp823 = mux(tmp299, UInt<1>(1), tmp822)
      node tmp824 = mux(tmp297, UInt<1>(1), tmp823)
      node tmp825 = mux(tmp295, UInt<1>(0), tmp824)
      node tmp826 = mux(tmp293, UInt<1>(0), tmp825)
      node tmp827 = mux(tmp291, UInt<1>(0), tmp826)
      node tmp828 = mux(tmp289, UInt<1>(0), tmp827)
      node tmp829 = mux(tmp287, UInt<1>(0), tmp828)
      node tmp830 = mux(tmp285, UInt<1>(0), tmp829)
      node tmp831 = mux(tmp283, UInt<1>(1), tmp830)
      node tmp832 = mux(tmp281, UInt<1>(1), tmp831)
      node tmp833 = mux(tmp279, UInt<1>(1), tmp832)
      node tmp834 = mux(tmp277, UInt<1>(1), tmp833)
      node tmp835 = mux(tmp361, UInt<2>(3), UInt<2>(0))
      node tmp836 = mux(tmp359, UInt<2>(2), tmp835)
      node tmp837 = mux(tmp357, UInt<2>(1), tmp836)
      node tmp838 = mux(tmp355, UInt<2>(3), tmp837)
      node tmp839 = mux(tmp353, UInt<2>(2), tmp838)
      node tmp840 = mux(tmp351, UInt<2>(1), tmp839)
      node tmp841 = mux(tmp349, UInt<2>(0), tmp840)
      node tmp842 = mux(tmp347, UInt<2>(0), tmp841)
      node tmp843 = mux(tmp345, UInt<2>(0), tmp842)
      node tmp844 = mux(tmp343, UInt<2>(0), tmp843)
      node tmp845 = mux(tmp341, UInt<2>(0), tmp844)
      node tmp846 = mux(tmp339, UInt<2>(0), tmp845)
      node tmp847 = mux(tmp337, UInt<2>(0), tmp846)
      node tmp848 = mux(tmp335, UInt<2>(0), tmp847)
      node tmp849 = mux(tmp333, UInt<2>(0), tmp848)
      node tmp850 = mux(tmp331, UInt<2>(0), tmp849)
      node tmp851 = mux(tmp329, UInt<2>(0), tmp850)
      node tmp852 = mux(tmp327, UInt<2>(0), tmp851)
      node tmp853 = mux(tmp325, UInt<2>(0), tmp852)
      node tmp854 = mux(tmp323, UInt<2>(0), tmp853)
      node tmp855 = mux(tmp321, UInt<2>(0), tmp854)
      node tmp856 = mux(tmp319, UInt<2>(0), tmp855)
      node tmp857 = mux(tmp317, UInt<2>(0), tmp856)
      node tmp858 = mux(tmp315, UInt<2>(0), tmp857)
      node tmp859 = mux(tmp313, UInt<2>(0), tmp858)
      node tmp860 = mux(tmp311, UInt<2>(0), tmp859)
      node tmp861 = mux(tmp309, UInt<2>(0), tmp860)
      node tmp862 = mux(tmp307, UInt<2>(0), tmp861)
      node tmp863 = mux(tmp305, UInt<2>(0), tmp862)
      node tmp864 = mux(tmp303, UInt<2>(0), tmp863)
      node tmp865 = mux(tmp301, UInt<2>(0), tmp864)
      node tmp866 = mux(tmp299, UInt<2>(0), tmp865)
      node tmp867 = mux(tmp297, UInt<2>(0), tmp866)
      node tmp868 = mux(tmp295, UInt<2>(0), tmp867)
      node tmp869 = mux(tmp293, UInt<2>(0), tmp868)
      node tmp870 = mux(tmp291, UInt<2>(0), tmp869)
      node tmp871 = mux(tmp289, UInt<2>(0), tmp870)
      node tmp872 = mux(tmp287, UInt<2>(0), tmp871)
      node tmp873 = mux(tmp285, UInt<2>(0), tmp872)
      node tmp874 = mux(tmp283, UInt<2>(0), tmp873)
      node tmp875 = mux(tmp281, UInt<2>(0), tmp874)
      node tmp876 = mux(tmp279, UInt<2>(0), tmp875)
      node tmp877 = mux(tmp277, UInt<2>(0), tmp876)
      node tmp878 = bits(ctrl.ins, 19, 15)
      node rs1_addr879 = tmp878
      node tmp880 = bits(ctrl.ins, 24, 20)
      node rs2_addr881 = tmp880
      reg st_type882 : UInt<1>
      st_type882 := ctrl.st_type
      reg ld_type883 : UInt<1>
      ld_type883 := tmp748
      reg wb_sel884 : UInt<1>
      wb_sel884 := tmp791
      reg wb_en885 : UInt<1>
      wb_en885 := tmp834
      reg csr_cmd886 : UInt<1>
      csr_cmd886 := tmp877
      ctrl.pc_sel := tmp404
      node tmp887 = bit-not(ctrl.data_re)
      node tmp888 = bit-and(ctrl.stall, tmp887)
      node tmp889 = bit-not(tmp888)
      ctrl.inst_re := tmp889
      node tmp890 = neq(tmp748, UInt<3>(7))
      wire tmp891 : UInt<1>
      tmp891 := tmp662
      node tmp892 = bit-or(tmp890, tmp891)
      node tmp893 = mux(tmp892, UInt<1>(1), UInt<1>(0))
      ctrl.inst_type := tmp893
      ctrl.A_sel := tmp447
      ctrl.B_sel := tmp490
      ctrl.imm_sel := tmp533
      ctrl.alu_op := tmp576
      ctrl.br_type := tmp619
      ctrl.st_type := tmp705
      node tmp894 = bit-not(ctrl.stall)
      when tmp894 :
         st_type882 := ctrl.st_type
         ld_type883 := tmp748
         wb_sel884 := tmp791
         wb_en885 := tmp834
         csr_cmd886 := tmp877      
      node tmp895 = neq(ctrl.ld_type, UInt<3>(7))
      node tmp896 = neq(tmp748, UInt<3>(7))
      node tmp897 = mux(ctrl.stall, tmp895, tmp896)
      ctrl.data_re := tmp897
      ctrl.ld_type := ld_type883
      ctrl.wb_en := wb_en885
      ctrl.wb_sel := wb_sel884
      ctrl.csr_cmd := csr_cmd886   
   module Core :
      input stall : UInt<1>
      output host : {flip hid : UInt<1>, tohost : UInt<32>, status : UInt<32>}
      output icache : {addr : UInt<32>, din : UInt<32>, flip dout : UInt<32>, re : UInt<1>, we : UInt<4>}
      output dcache : {addr : UInt<32>, din : UInt<32>, flip dout : UInt<32>, re : UInt<1>, we : UInt<4>}
      inst dpath of Datapath
      inst ctrl of Control
      host := dpath.host
      icache := dpath.icache
      dcache := dpath.dcache
      dpath.ctrl := ctrl.ctrl
      dpath.stall := stall   
   module Fifo :
      input enq : {valid : UInt<1>, bits : {rw : UInt<1>, addr : UInt<32>, tag : UInt<5>, mask : UInt<4>}, flip ready : UInt<1>}
      output deq : {valid : UInt<1>, bits : {rw : UInt<1>, addr : UInt<32>, tag : UInt<5>, mask : UInt<4>}, flip ready : UInt<1>}
      output count : UInt<3>
      mem ram : {rw : UInt<1>, addr : UInt<32>, tag : UInt<5>, mask : UInt<4>}[4]
      reg cnt898 : UInt<2>
      on-reset cnt898 := UInt<1>(0)
      reg cnt899 : UInt<2>
      on-reset cnt899 := UInt<1>(0)
      reg maybe-full900 : UInt<1>
      on-reset maybe-full900 := UInt<1>(0)
      node tmp901 = eq(cnt898, cnt899)
      node ptr-match902 = tmp901
      node tmp903 = bit-not(maybe-full900)
      node tmp904 = bit-and(ptr-match902, tmp903)
      node empty905 = tmp904
      node tmp906 = bit-and(ptr-match902, maybe-full900)
      node full907 = tmp906
      node tmp908 = bit-and(UInt<1>(0), empty905)
      node maybe-flow909 = tmp908
      node tmp910 = bit-and(maybe-flow909, deq.ready)
      node do-flow911 = tmp910
      node tmp912 = bit-not(do-flow911)
      node tmp913 = bit-and(enq.valid, tmp912)
      node tmp914 = bit-and(enq.ready, tmp913)
      node do-enq915 = tmp914
      node tmp916 = bit-not(do-flow911)
      node tmp917 = bit-and(deq.valid, tmp916)
      node tmp918 = bit-and(deq.ready, tmp917)
      node do-deq919 = tmp918
      when do-enq915 :
         accessor a920 = ram[cnt898]
         a920 := enq.bits
         node tmp921 = eq(cnt898, UInt<2>(3))
         node wrap922 = tmp921
         node tmp923 = bit-and(UInt<1>(0), wrap922)
         node tmp924 = add-wrap(cnt898, UInt<1>(1))
         node tmp925 = mux(tmp923, UInt<1>(0), tmp924)
         cnt898 := tmp925      
      when do-deq919 :
         node tmp926 = eq(cnt899, UInt<2>(3))
         node wrap927 = tmp926
         node tmp928 = bit-and(UInt<1>(0), wrap927)
         node tmp929 = add-wrap(cnt899, UInt<1>(1))
         node tmp930 = mux(tmp928, UInt<1>(0), tmp929)
         cnt899 := tmp930      
      node tmp931 = neq(do-enq915, do-deq919)
      when tmp931 :
         maybe-full900 := do-enq915      
      node tmp932 = bit-and(UInt<1>(0), enq.valid)
      node tmp933 = bit-or(empty905, tmp932)
      node tmp934 = bit-not(tmp933)
      deq.valid := tmp934
      node tmp935 = bit-and(UInt<1>(0), deq.ready)
      node tmp936 = bit-or(full907, tmp935)
      node tmp937 = bit-not(tmp936)
      enq.ready := tmp937
      accessor a938 = ram[cnt899]
      node tmp939 = mux(maybe-flow909, enq.bits, a938)
      deq.bits := tmp939
      node tmp940 = sub-wrap(cnt898, cnt899)
      node ptr-diff941 = tmp940
      node tmp942 = bit-and(maybe-full900, ptr-match902)
      node tmp943 = cat(tmp942, ptr-diff941)
      count := tmp943   
   module Fifo944 :
      input enq : {valid : UInt<1>, bits : {data : UInt<32>}, flip ready : UInt<1>}
      output deq : {valid : UInt<1>, bits : {data : UInt<32>}, flip ready : UInt<1>}
      output count : UInt<3>
      mem ram : {data : UInt<32>}[4]
      reg cnt945 : UInt<2>
      on-reset cnt945 := UInt<1>(0)
      reg cnt946 : UInt<2>
      on-reset cnt946 := UInt<1>(0)
      reg maybe-full947 : UInt<1>
      on-reset maybe-full947 := UInt<1>(0)
      node tmp948 = eq(cnt945, cnt946)
      node ptr-match949 = tmp948
      node tmp950 = bit-not(maybe-full947)
      node tmp951 = bit-and(ptr-match949, tmp950)
      node empty952 = tmp951
      node tmp953 = bit-and(ptr-match949, maybe-full947)
      node full954 = tmp953
      node tmp955 = bit-and(UInt<1>(0), empty952)
      node maybe-flow956 = tmp955
      node tmp957 = bit-and(maybe-flow956, deq.ready)
      node do-flow958 = tmp957
      node tmp959 = bit-not(do-flow958)
      node tmp960 = bit-and(enq.valid, tmp959)
      node tmp961 = bit-and(enq.ready, tmp960)
      node do-enq962 = tmp961
      node tmp963 = bit-not(do-flow958)
      node tmp964 = bit-and(deq.valid, tmp963)
      node tmp965 = bit-and(deq.ready, tmp964)
      node do-deq966 = tmp965
      when do-enq962 :
         accessor a967 = ram[cnt945]
         a967 := enq.bits
         node tmp968 = eq(cnt945, UInt<2>(3))
         node wrap969 = tmp968
         node tmp970 = bit-and(UInt<1>(0), wrap969)
         node tmp971 = add-wrap(cnt945, UInt<1>(1))
         node tmp972 = mux(tmp970, UInt<1>(0), tmp971)
         cnt945 := tmp972      
      when do-deq966 :
         node tmp973 = eq(cnt946, UInt<2>(3))
         node wrap974 = tmp973
         node tmp975 = bit-and(UInt<1>(0), wrap974)
         node tmp976 = add-wrap(cnt946, UInt<1>(1))
         node tmp977 = mux(tmp975, UInt<1>(0), tmp976)
         cnt946 := tmp977      
      node tmp978 = neq(do-enq962, do-deq966)
      when tmp978 :
         maybe-full947 := do-enq962      
      node tmp979 = bit-and(UInt<1>(0), enq.valid)
      node tmp980 = bit-or(empty952, tmp979)
      node tmp981 = bit-not(tmp980)
      deq.valid := tmp981
      node tmp982 = bit-and(UInt<1>(0), deq.ready)
      node tmp983 = bit-or(full954, tmp982)
      node tmp984 = bit-not(tmp983)
      enq.ready := tmp984
      accessor a985 = ram[cnt946]
      node tmp986 = mux(maybe-flow956, enq.bits, a985)
      deq.bits := tmp986
      node tmp987 = sub-wrap(cnt945, cnt946)
      node ptr-diff988 = tmp987
      node tmp989 = bit-and(maybe-full947, ptr-match949)
      node tmp990 = cat(tmp989, ptr-diff988)
      count := tmp990   
   module Memory :
      output stall : UInt<1>
      output memory : {req_cmd : {valid : UInt<1>, bits : {rw : UInt<1>, addr : UInt<32>, tag : UInt<5>, mask : UInt<4>}, flip ready : UInt<1>}, req_data : {valid : UInt<1>, bits : {data : UInt<32>}, flip ready : UInt<1>}, flip resp : {valid : UInt<1>, bits : {data : UInt<32>, tag : UInt<5>}, flip ready : UInt<1>}}
      input icache : {addr : UInt<32>, din : UInt<32>, flip dout : UInt<32>, re : UInt<1>, we : UInt<4>}
      input dcache : {addr : UInt<32>, din : UInt<32>, flip dout : UInt<32>, re : UInt<1>, we : UInt<4>}
      inst memReqCmdQueue of Fifo
      inst memReqDataQueue of Fifo944
      reg state991 : UInt<?>
      on-reset state991 := UInt<1>(0)
      reg tag992 : UInt<5>
      on-reset tag992 := UInt<1>(0)
      node tmp993 = eq(state991, UInt<1>(0))
      node tmp994 = eq(dcache.we, UInt<1>(0))
      node tmp995 = bit-not(tmp994)
      node tmp996 = bit-or(dcache.re, tmp995)
      node tmp997 = bit-or(icache.re, tmp996)
      node tmp998 = bit-and(tmp993, tmp997)
      node cpuReq999 = tmp998
      node tmp1000 = bits(icache.addr, 31, 2)
      node tmp1001 = cat(tmp1000, UInt<2>(0))
      node iaddr1002 = tmp1001
      node tmp1003 = bits(dcache.addr, 31, 2)
      node tmp1004 = cat(tmp1003, UInt<2>(0))
      node daddr1005 = tmp1004
      reg idata1006 : UInt<?>
      reg ddata1007 : UInt<?>
      reg ire1008 : UInt<1>
      reg dre1009 : UInt<1>
      icache.dout := idata1006
      dcache.dout := ddata1007
      memory.req_cmd := memReqCmdQueue.deq
      memory.req_data := memReqDataQueue.deq
      memory.resp.ready := UInt<1>(0)
      node tmp1010 = eq(state991, UInt<1>(1))
      node tmp1011 = bit-not(memReqDataQueue.enq.ready)
      node tmp1012 = bit-or(memReqCmdQueue.enq.ready, tmp1011)
      node tmp1013 = bit-not(tmp1012)
      node tmp1014 = bit-or(tmp1010, tmp1013)
      stall := tmp1014
      node tmp1015 = eq(dcache.we, UInt<1>(0))
      node tmp1016 = bit-not(tmp1015)
      memReqCmdQueue.enq.bits.rw := tmp1016
      memReqCmdQueue.enq.bits.tag := tag992
      node tmp1017 = eq(dcache.we, UInt<1>(0))
      node tmp1018 = bit-not(tmp1017)
      node tmp1019 = bit-not(icache.re)
      node tmp1020 = bit-or(tmp1018, tmp1019)
      node tmp1021 = mux(tmp1020, daddr1005, iaddr1002)
      memReqCmdQueue.enq.bits.addr := tmp1021
      memReqCmdQueue.enq.bits.mask := dcache.we
      node tmp1022 = bit-and(memReqDataQueue.enq.ready, cpuReq999)
      memReqCmdQueue.enq.valid := tmp1022
      memReqDataQueue.enq.bits.data := dcache.din
      node tmp1023 = eq(dcache.we, UInt<1>(0))
      node tmp1024 = bit-not(tmp1023)
      node tmp1025 = bit-and(cpuReq999, tmp1024)
      node tmp1026 = bit-and(memReqCmdQueue.enq.ready, tmp1025)
      memReqDataQueue.enq.valid := tmp1026
      node tmp1027 = eq(state991, UInt<1>(0))
      when tmp1027 :
         node tmp1028 = bit-or(icache.re, dcache.re)
         node tmp1029 = eq(dcache.we, UInt<1>(0))
         node tmp1030 = bit-not(tmp1029)
         node tmp1031 = bit-and(tmp1030, memReqCmdQueue.enq.ready)
         node tmp1032 = bit-not(tmp1031)
         node tmp1033 = bit-and(tmp1028, tmp1032)
         when tmp1033 :
            ire1008 := icache.re
            dre1009 := dcache.re
            state991 := UInt<1>(1)      
      node tmp1034 = eq(state991, UInt<1>(1))
      when tmp1034 :
         memory.resp.ready := UInt<1>(1)
         node tmp1035 = eq(memory.resp.bits.tag, tag992)
         node tmp1036 = bit-and(memory.resp.valid, tmp1035)
         when tmp1036 :
            state991 := UInt<1>(0)
            node tmp1037 = add-wrap(tag992, UInt<1>(1))
            tag992 := tmp1037
            memory.resp.ready := UInt<1>(0)
            when ire1008 :
               idata1006 := memory.resp.bits.data            
            when dre1009 :
               ddata1007 := memory.resp.bits.data   
   module Tile :
      output htif : {host : {flip hid : UInt<1>, tohost : UInt<32>, status : UInt<32>}}
      output memory : {req_cmd : {valid : UInt<1>, bits : {rw : UInt<1>, addr : UInt<32>, tag : UInt<5>, mask : UInt<4>}, flip ready : UInt<1>}, req_data : {valid : UInt<1>, bits : {data : UInt<32>}, flip ready : UInt<1>}, flip resp : {valid : UInt<1>, bits : {data : UInt<32>, tag : UInt<5>}, flip ready : UInt<1>}}
      inst core of Core
      inst memmod of Memory
      htif.host := core.host
      memory := memmod.memory
      core.stall := memmod.stall
      memmod.icache := core.icache
      memmod.dcache := core.dcache   

