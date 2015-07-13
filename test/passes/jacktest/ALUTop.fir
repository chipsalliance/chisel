; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
;CHECK: Done!
circuit ALUTop : 
  module ALU : 
    input B : UInt<32>
    output out : UInt<32>
    output sum : UInt<32>
    input A : UInt<32>
    input alu_op : UInt<4>
    
    node shamt = bits(B, 4, 0)
    node T_157 = addw(A, B)
    node T_158 = subw(A, B)
    node T_159 = cvt(A)
    node T_160 = dshr(T_159, shamt)
    node T_161 = asUInt(T_160)
    node T_162 = dshr(A, shamt)
    node T_163 = dshl(A, shamt)
    node T_164 = bits(T_163, 31, 0)
    node T_165 = cvt(A)
    node T_166 = cvt(B)
    node T_167 = lt(T_165, T_166)
    node T_168 = asUInt(T_167)
    node T_169 = lt(A, B)
    node T_170 = asUInt(T_169)
    node T_171 = and(A, B)
    node T_172 = or(A, B)
    node T_173 = xor(A, B)
    node T_174 = eq(UInt<4>(10), alu_op)
    node T_175 = mux(T_174, A, B)
    node T_176 = eq(UInt<4>(4), alu_op)
    node T_177 = mux(T_176, T_173, T_175)
    node T_178 = eq(UInt<4>(3), alu_op)
    node T_179 = mux(T_178, T_172, T_177)
    node T_180 = eq(UInt<4>(2), alu_op)
    node T_181 = mux(T_180, T_171, T_179)
    node T_182 = eq(UInt<4>(7), alu_op)
    node T_183 = mux(T_182, T_170, T_181)
    node T_184 = eq(UInt<4>(5), alu_op)
    node T_185 = mux(T_184, T_168, T_183)
    node T_186 = eq(UInt<4>(6), alu_op)
    node T_187 = mux(T_186, T_164, T_185)
    node T_188 = eq(UInt<4>(8), alu_op)
    node T_189 = mux(T_188, T_162, T_187)
    node T_190 = eq(UInt<4>(9), alu_op)
    node T_191 = mux(T_190, T_161, T_189)
    node T_192 = eq(UInt<4>(1), alu_op)
    node T_193 = mux(T_192, T_158, T_191)
    node T_194 = eq(UInt<4>(0), alu_op)
    node oot = mux(T_194, T_157, T_193)
    node T_195 = bits(oot, 31, 0)
    out := T_195
    node T_196 = bit(alu_op, 0)
    node T_197 = subw(UInt<1>(0), B)
    node T_198 = mux(T_196, T_197, B)
    node T_199 = addw(A, T_198)
    sum := T_199
  module ALUdec : 
    input opcode : UInt<7>
    input funct : UInt<3>
    input add_rshift_type : UInt<1>
    output alu_op : UInt<4>
    
    node T_200 = mux(add_rshift_type, UInt<4>(1), UInt<4>(0))
    node T_201 = mux(add_rshift_type, UInt<4>(9), UInt<4>(8))
    node T_202 = eq(UInt<3>(5), funct)
    node T_203 = mux(T_202, T_201, UInt<4>(15))
    node T_204 = eq(UInt<3>(7), funct)
    node T_205 = mux(T_204, UInt<4>(2), T_203)
    node T_206 = eq(UInt<3>(6), funct)
    node T_207 = mux(T_206, UInt<4>(3), T_205)
    node T_208 = eq(UInt<3>(4), funct)
    node T_209 = mux(T_208, UInt<4>(4), T_207)
    node T_210 = eq(UInt<3>(3), funct)
    node T_211 = mux(T_210, UInt<4>(7), T_209)
    node T_212 = eq(UInt<3>(2), funct)
    node T_213 = mux(T_212, UInt<4>(5), T_211)
    node T_214 = eq(UInt<3>(1), funct)
    node T_215 = mux(T_214, UInt<4>(6), T_213)
    node T_216 = eq(UInt<3>(0), funct)
    node alu_op1 = mux(T_216, T_200, T_215)
    node T_217 = eq(UInt<7>(19), opcode)
    node T_218 = mux(T_217, alu_op1, UInt<4>(15))
    node T_219 = eq(UInt<7>(51), opcode)
    node T_220 = mux(T_219, alu_op1, T_218)
    node T_221 = eq(UInt<7>(3), opcode)
    node T_222 = mux(T_221, UInt<4>(0), T_220)
    node T_223 = eq(UInt<7>(35), opcode)
    node T_224 = mux(T_223, UInt<4>(0), T_222)
    node T_225 = eq(UInt<7>(99), opcode)
    node T_226 = mux(T_225, UInt<4>(0), T_224)
    node T_227 = eq(UInt<7>(103), opcode)
    node T_228 = mux(T_227, UInt<4>(0), T_226)
    node T_229 = eq(UInt<7>(111), opcode)
    node T_230 = mux(T_229, UInt<4>(0), T_228)
    node T_231 = eq(UInt<7>(23), opcode)
    node T_232 = mux(T_231, UInt<4>(0), T_230)
    node T_233 = eq(UInt<7>(55), opcode)
    node alu_op2 = mux(T_233, UInt<4>(11), T_232)
    alu_op := alu_op2
  module ALUTop : 
    input B : UInt<32>
    output out : UInt<32>
    input A : UInt<32>
    input opcode : UInt<7>
    input funct : UInt<3>
    input add_rshift_type : UInt<1>
    
    inst alu of ALU
    inst alu_dec of ALUdec
    alu_dec.opcode := opcode
    alu_dec.funct := funct
    alu_dec.add_rshift_type := add_rshift_type
    alu.A := A
    alu.B := B
    out := alu.out
    alu.alu_op := alu_dec.alu_op
