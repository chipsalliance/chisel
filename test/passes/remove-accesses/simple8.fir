; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

circuit top :
  module top : 
    output resp : UInt<128>[4]
    input write : {flip ready : UInt<1>, valid : UInt<1>, bits : {way_en : UInt<4>, addr : UInt<12>, wmask : UInt<2>, data : UInt<128>}}
    input read : {flip ready : UInt<1>, valid : UInt<1>, bits : {way_en : UInt<4>, addr : UInt<12>}}
    input clock : Clock
    input reset : UInt<1>
    
    resp[0] <= UInt<1>("h00")
    resp[1] <= UInt<1>("h00")
    resp[2] <= UInt<1>("h00")
    resp[3] <= UInt<1>("h00")
    write.ready <= UInt<1>("h00")
    read.ready <= UInt<1>("h00")
    node waddr = shr(write.bits.addr, 4)
    node raddr = shr(read.bits.addr, 4)
    node T_65 = bits(write.bits.way_en, 1, 0)
    node T_66 = bits(read.bits.way_en, 1, 0)
    wire T_75 : UInt<128>[2]
    T_75[0] <= UInt<1>("h00")
    T_75[1] <= UInt<1>("h00")
    reg T_81 : UInt<12>, clock, reset, T_81
    when read.valid :
      T_81 <= read.bits.addr
      skip
    cmem T_84 : UInt<128>[256]
    node T_86 = neq(T_65, UInt<1>("h00"))
    node T_87 = and(T_86, write.valid)
    node T_88 = bit(write.bits.wmask, 0)
    node T_89 = and(T_87, T_88)
    when T_89 :
      node T_90 = bits(write.bits.data, 63, 0)
      node T_91 = cat(T_90, T_90)
      node T_92 = bit(T_65, 0)
      node T_93 = bit(T_65, 1)
      wire T_95 : UInt<1>[2]
      T_95[0] <= T_92
      T_95[1] <= T_93
      node T_100 = subw(UInt<64>("h00"), T_95[0])
      node T_102 = subw(UInt<64>("h00"), T_95[1])
      wire T_104 : UInt<64>[2]
      T_104[0] <= T_100
      T_104[1] <= T_102
      node T_108 = cat(T_104[1], T_104[0])
      read mport T_109 = T_84[waddr],clock
      node T_110 = not(T_108)
      node T_111 = and(T_109, T_110)
      node T_112 = and(T_91, T_108)
      node T_113 = or(T_111, T_112)
      wire T_114 : UInt<128>
      T_114 <= UInt<1>("h00")
      T_114 <= T_113
      write mport T_116 = T_84[waddr],clock,UInt(1)
      T_116 <= T_114
      skip
    node T_118 = neq(T_66, UInt<1>("h00"))
    node T_119 = and(T_118, read.valid)
    reg T_120 : UInt<8>, clock, reset, T_120
    when T_119 :
      T_120 <= raddr
      skip
    read mport T_121 = T_84[T_120],clock
    T_75[0] <= T_121
    cmem T_124 : UInt<128>[256]
    node T_126 = neq(T_65, UInt<1>("h00"))
    node T_127 = and(T_126, write.valid)
    node T_128 = bit(write.bits.wmask, 1)
    node T_129 = and(T_127, T_128)
    when T_129 :
      node T_130 = bits(write.bits.data, 127, 64)
      node T_131 = cat(T_130, T_130)
      node T_132 = bit(T_65, 0)
      node T_133 = bit(T_65, 1)
      wire T_135 : UInt<1>[2]
      T_135[0] <= T_132
      T_135[1] <= T_133
      node T_140 = subw(UInt<64>("h00"), T_135[0])
      node T_142 = subw(UInt<64>("h00"), T_135[1])
      wire T_144 : UInt<64>[2]
      T_144[0] <= T_140
      T_144[1] <= T_142
      node T_148 = cat(T_144[1], T_144[0])
      read mport T_149 = T_124[waddr],clock
      node T_150 = not(T_148)
      node T_151 = and(T_149, T_150)
      node T_152 = and(T_131, T_148)
      node T_153 = or(T_151, T_152)
      wire T_154 : UInt<128>
      T_154 <= UInt<1>("h00")
      T_154 <= T_153
      write mport T_156 = T_124[waddr],clock,UInt(1)
      T_156 <= T_154
      skip
    node T_158 = neq(T_66, UInt<1>("h00"))
    node T_159 = and(T_158, read.valid)
    reg T_160 : UInt<8>, clock, reset, T_160
    when T_159 :
      T_160 <= raddr
      skip
    read mport T_161 = T_124[T_160],clock
    T_75[1] <= T_161
    node T_162 = bits(T_75[0], 63, 0)
    node T_163 = bits(T_75[1], 63, 0)
    wire T_165 : UInt<64>[2]
    T_165[0] <= T_162
    T_165[1] <= T_163
    node T_169 = bits(T_81, 3, 3)
    wire T_172 : UInt<64>[2]
    T_172[0] <= T_165[T_169]
    T_172[1] <= T_165[1]
    node T_176 = cat(T_172[1], T_172[0])
    resp[0] <= T_176
    node T_177 = bits(T_75[0], 127, 64)
    node T_178 = bits(T_75[1], 127, 64)
    wire T_180 : UInt<64>[2]
    T_180[0] <= T_177
    T_180[1] <= T_178
    node T_184 = bits(T_81, 3, 3)
    wire T_187 : UInt<64>[2]
    T_187[0] <= T_180[T_184]
    T_187[1] <= T_180[1]
    node T_191 = cat(T_187[1], T_187[0])
    resp[1] <= T_191
    node T_192 = bits(write.bits.way_en, 3, 2)
    node T_193 = bits(read.bits.way_en, 3, 2)
    wire T_202 : UInt<128>[2]
    T_202[0] <= UInt<1>("h00")
    T_202[1] <= UInt<1>("h00")
    reg T_208 : UInt<12>, clock, reset, T_208
    when read.valid :
      T_208 <= read.bits.addr
      skip
    cmem T_211 : UInt<128>[256]
    node T_213 = neq(T_192, UInt<1>("h00"))
    node T_214 = and(T_213, write.valid)
    node T_215 = bit(write.bits.wmask, 0)
    node T_216 = and(T_214, T_215)
    when T_216 :
      node T_217 = bits(write.bits.data, 63, 0)
      node T_218 = cat(T_217, T_217)
      node T_219 = bit(T_192, 0)
      node T_220 = bit(T_192, 1)
      wire T_222 : UInt<1>[2]
      T_222[0] <= T_219
      T_222[1] <= T_220
      node T_227 = subw(UInt<64>("h00"), T_222[0])
      node T_229 = subw(UInt<64>("h00"), T_222[1])
      wire T_231 : UInt<64>[2]
      T_231[0] <= T_227
      T_231[1] <= T_229
      node T_235 = cat(T_231[1], T_231[0])
      read mport T_236 = T_211[waddr],clock
      node T_237 = not(T_235)
      node T_238 = and(T_236, T_237)
      node T_239 = and(T_218, T_235)
      node T_240 = or(T_238, T_239)
      wire T_241 : UInt<128>
      T_241 <= UInt<1>("h00")
      T_241 <= T_240
      write mport T_243 = T_211[waddr],clock,UInt(1)
      T_243 <= T_241
      skip
    node T_245 = neq(T_193, UInt<1>("h00"))
    node T_246 = and(T_245, read.valid)
    reg T_247 : UInt<8>, clock, reset, T_247
    when T_246 :
      T_247 <= raddr
      skip
    read mport T_248 = T_211[T_247],clock
    T_202[0] <= T_248
    cmem T_251 : UInt<128>[256]
    node T_253 = neq(T_192, UInt<1>("h00"))
    node T_254 = and(T_253, write.valid)
    node T_255 = bit(write.bits.wmask, 1)
    node T_256 = and(T_254, T_255)
    when T_256 :
      node T_257 = bits(write.bits.data, 127, 64)
      node T_258 = cat(T_257, T_257)
      node T_259 = bit(T_192, 0)
      node T_260 = bit(T_192, 1)
      wire T_262 : UInt<1>[2]
      T_262[0] <= T_259
      T_262[1] <= T_260
      node T_267 = subw(UInt<64>("h00"), T_262[0])
      node T_269 = subw(UInt<64>("h00"), T_262[1])
      wire T_271 : UInt<64>[2]
      T_271[0] <= T_267
      T_271[1] <= T_269
      node T_275 = cat(T_271[1], T_271[0])
      read mport T_276 = T_251[waddr],clock
      node T_277 = not(T_275)
      node T_278 = and(T_276, T_277)
      node T_279 = and(T_258, T_275)
      node T_280 = or(T_278, T_279)
      wire T_281 : UInt<128>
      T_281 <= UInt<1>("h00")
      T_281 <= T_280
      write mport T_283 = T_251[waddr],clock,UInt(1)
      T_283 <= T_281
      skip
    node T_285 = neq(T_193, UInt<1>("h00"))
    node T_286 = and(T_285, read.valid)
    reg T_287 : UInt<8>, clock, reset, T_287
    when T_286 :
      T_287 <= raddr
      skip
    read mport T_288 = T_251[T_287],clock
    T_202[1] <= T_288
    node T_289 = bits(T_202[0], 63, 0)
    node T_290 = bits(T_202[1], 63, 0)
    wire T_292 : UInt<64>[2]
    T_292[0] <= T_289
    T_292[1] <= T_290
    node T_296 = bits(T_208, 3, 3)
    wire T_299 : UInt<64>[2]
    T_299[0] <= T_292[T_296]
    T_299[1] <= T_292[1]
    node T_303 = cat(T_299[1], T_299[0])
    resp[2] <= T_303
    node T_304 = bits(T_202[0], 127, 64)
    node T_305 = bits(T_202[1], 127, 64)
    wire T_307 : UInt<64>[2]
    T_307[0] <= T_304
    T_307[1] <= T_305
    node T_311 = bits(T_208, 3, 3)
    wire T_314 : UInt<64>[2]
    T_314[0] <= T_307[T_311]
    T_314[1] <= T_307[1]
    node T_318 = cat(T_314[1], T_314[0])
    resp[3] <= T_318
    read.ready <= UInt<1>("h01")
    write.ready <= UInt<1>("h01")
;CHECK: Done!
