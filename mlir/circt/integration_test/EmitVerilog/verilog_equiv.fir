; REQUIRES: yosys
; RUN: split-file %s %t
; RUN: firtool %t/test_mod.fir --format=fir -verilog -o %t/test_mod.fir.v
; RUN: firtool %t/test_unary.fir --format=fir -verilog -o %t/test_unary.fir.v
; RUN: firtool %t/test_prim.fir --format=fir -verilog -o %t/test_prim.fir.v
; RUN: equiv-rtl.sh %t/test_mod.fir.v %t/test_mod.v test_mod
; RUN: equiv-rtl.sh %t/test_unary.fir.v %t/test_unary.v test_unary
; RUN: equiv-rtl.sh %t/test_prim.fir.v %t/test_prim.v test_prim

;--- test_mod.fir

circuit test_mod :
  module test_mod :
    input a: UInt<1>
    output b: UInt<1>
    b <= a

;--- test_mod.v

module test_mod(
  input   a,
  output  b
);
  assign b = a;
endmodule

;--- test_unary.fir

circuit test_unary :
  module test_unary :
    input uin4: UInt<4>
    input sin4: SInt<4>
    output out_xorr_u: UInt<1>
    output out_xorr_s: UInt<1>
    output out_andr_u: UInt<1>
    output out_andr_s: UInt<1>
    output out_orr_u: UInt<1>
    output out_orr_s: UInt<1>
    output out_not_u: UInt<4>
    output out_not_s: UInt<4>
    output out_pad_u: UInt<6>
    output out_pad_s: SInt<6>
    output out_shl_u: UInt<6>
    output out_shl_s: SInt<6>
    output out_shr_u: UInt<2>
    output out_shr_s: SInt<2>
    output out_cvt_u: SInt<5>
    output out_cvt_s: SInt<4>
    output out_bits_u: UInt<3>
    output out_bits_s: UInt<3>
    output out_head_u: UInt<2>
    output out_head_s: UInt<2>
    output out_tail_u: UInt<2>
    output out_tail_s: UInt<2>
    output out_neg_s: SInt<5>
    output out_neg_u: SInt<5>
    out_xorr_u <= xorr(uin4)
    out_xorr_s <= xorr(sin4)
    out_andr_u <= andr(uin4)
    out_andr_s <= andr(sin4)
    out_orr_u <= orr(uin4)
    out_orr_s <= orr(sin4)
    out_not_u <= not(uin4)
    out_not_s <= not(sin4)
    out_pad_u <= pad(uin4, 6)
    out_pad_s <= pad(sin4, 6)
    out_shl_u <= shl(uin4, 2)
    out_shl_s <= shl(sin4, 2)
    out_shr_u <= shr(uin4, 2)
    out_shr_s <= shr(sin4, 2)
    out_cvt_u <= cvt(uin4)
    out_cvt_s <= cvt(sin4)
    out_bits_u <= bits(uin4, 3,1)
    out_bits_s <= bits(sin4, 3,1)
    out_head_u <= head(uin4, 2)
    out_head_s <= head(sin4, 2)
    out_tail_u <= tail(uin4, 2)
    out_tail_s <= tail(sin4, 2)
    out_neg_s <= neg(sin4)
    out_neg_u <= neg(uin4)
    
;--- test_unary.v

module test_unary(
  input  [3:0] uin4,
  input  [3:0] sin4,
  output       out_xorr_u,
  output       out_xorr_s,
  output       out_andr_u,
  output       out_andr_s,
  output       out_orr_u,
  output       out_orr_s,
  output [3:0] out_not_u,
  output [3:0] out_not_s,
  output [5:0] out_pad_u,
  output [5:0] out_pad_s,
  output [5:0] out_shl_u,
  output [5:0] out_shl_s,
  output [1:0] out_shr_u,
  output [1:0] out_shr_s,
  output [4:0] out_cvt_u,
  output [3:0] out_cvt_s,
  output [2:0] out_bits_u,
  output [2:0] out_bits_s,
  output [1:0] out_head_u,
  output [1:0] out_head_s,
  output [1:0] out_tail_u,
  output [1:0] out_tail_s,
  output [4:0] out_neg_u,
  output [4:0] out_neg_s
);
  assign out_xorr_u = ^uin4;
  assign out_xorr_s = ^sin4;
  assign out_andr_u = &uin4;
  assign out_andr_s = &sin4;
  assign out_orr_u = |uin4;
  assign out_orr_s = |sin4;
  assign out_not_u = ~uin4;
  assign out_not_s = ~sin4;
  assign out_pad_u = {{2'd0}, uin4};
  assign out_pad_s = {{2{sin4[3]}},sin4};
  assign out_shl_u = {uin4, 2'h0};
  assign out_shl_s = {$signed(sin4), 2'h0};
  assign out_shr_u = uin4[3:2];
  assign out_shr_s = sin4[3:2];
  assign out_cvt_u = {1'b0,$signed(uin4)};
  assign out_cvt_s = $signed(sin4);
  assign out_bits_u = uin4[3:1];
  assign out_bits_s = sin4[3:1];
  assign out_head_u = uin4[3:2];
  assign out_head_s = sin4[3:2];
  assign out_tail_u = uin4[1:0];
  assign out_tail_s = sin4[1:0];
  assign out_neg_s = 4'sh0 - $signed(sin4); 
  assign out_neg_u = 4'h0 - uin4;
endmodule

;--- test_prim.fir

circuit test_prim :
  module test_prim :
    input uina4: UInt<4>
    input uinb4: UInt<4>
    input sina4: SInt<4>
    input sinb4: SInt<4>
    output out_add_u: UInt<5>
    output out_add_s: SInt<5>
    output out_sub_u: UInt<5>
    output out_sub_s: SInt<5>
    output out_mul_u: UInt<8>
    output out_mul_s: SInt<8>
    output out_div_u: UInt<4>
    output out_div_s: SInt<5>
    output out_rem_u: UInt<4>
    output out_rem_s: SInt<4>
    output out_lt_u: UInt<1>
    output out_lt_s: UInt<1>
    output out_leq_u: UInt<1>
    output out_leq_s: UInt<1>
    output out_gt_u: UInt<1>
    output out_gt_s: UInt<1>
    output out_geq_u: UInt<1>
    output out_geq_s: UInt<1>
    output out_eq_u: UInt<1>
    output out_eq_s: UInt<1>
    output out_neq_u: UInt<1>
    output out_neq_s: UInt<1>
    output out_dshl_u: UInt<19>
    output out_dshl_s: SInt<19>
    output out_dshr_u: UInt<4>
    output out_dshr_s: SInt<4>
    output out_and_u: UInt<4>
    output out_and_s: UInt<4>
    output out_or_u: UInt<4>
    output out_or_s: UInt<4>
    output out_xor_u: UInt<4>
    output out_xor_s: UInt<4>
    output out_cat_u: UInt<8>
    output out_cat_s: UInt<8>
    out_add_u <= add(uina4, uinb4)
    out_add_s <= add(sina4, sinb4)
    out_sub_u <= sub(uina4, uinb4)
    out_sub_s <= sub(sina4, sinb4)
    out_mul_u <= mul(uina4, uinb4)
    out_mul_s <= mul(sina4, sinb4)
    out_div_u <= div(uina4, uinb4)
    out_div_s <= div(sina4, sinb4)
    out_rem_u <= rem(uina4, uinb4)
    out_rem_s <= rem(sina4, sinb4)
    out_lt_u <= lt(uina4, uinb4)
    out_lt_s <= lt(sina4, sinb4)
    out_leq_u <= leq(uina4, uinb4)
    out_leq_s <= leq(sina4, sinb4)
    out_gt_u <= gt(uina4, uinb4)
    out_gt_s <= gt(sina4, sinb4)
    out_geq_u <= geq(uina4, uinb4)
    out_geq_s <= geq(sina4, sinb4)
    out_eq_u <= eq(uina4, uinb4)
    out_eq_s <= eq(sina4, sinb4)
    out_neq_u <= neq(uina4, uinb4)
    out_neq_s <= neq(sina4, sinb4)
    out_dshl_u <= dshl(uina4, uinb4)
    out_dshl_s <= dshl(sina4, uinb4)
    out_dshr_u <= dshr(uina4, uinb4)
    out_dshr_s <= dshr(sina4, uinb4)
    out_and_u <= and(uina4, uinb4)
    out_and_s <= and(sina4, sinb4)
    out_or_u <= or(uina4, uinb4)
    out_or_s <= or(sina4, sinb4)
    out_xor_u <= xor(uina4, uinb4)
    out_xor_s <= xor(sina4, sinb4)
    out_cat_u <= cat(uina4, uinb4)
    out_cat_s <= cat(sina4, sinb4)
    
;--- test_prim.v

module test_prim(
  input  [3:0] uina4,
  input  [3:0] uinb4,
  input  [3:0] sina4,
  input  [3:0] sinb4,
  output [4:0] out_add_u,
  output [4:0] out_add_s,
  output [4:0] out_sub_u,
  output [4:0] out_sub_s,
  output [7:0] out_mul_u,
  output [7:0] out_mul_s,
  output [3:0] out_div_u,
  output [4:0] out_div_s,
  output [3:0] out_rem_u,
  output [3:0] out_rem_s,
  output       out_lt_u,
  output       out_lt_s,
  output       out_leq_u,
  output       out_leq_s,
  output       out_gt_u,
  output       out_gt_s,
  output       out_geq_u,
  output       out_geq_s,
  output       out_eq_u,
  output       out_eq_s,
  output       out_neq_u,
  output       out_neq_s,
  output [18:0] out_dshl_u,
  output [18:0] out_dshl_s,
  output [3:0] out_dshr_u,
  output [3:0] out_dshr_s,
  output [3:0] out_and_u,
  output [3:0] out_and_s,
  output [3:0] out_or_u,
  output [3:0] out_or_s,
  output [3:0] out_xor_u,
  output [3:0] out_xor_s,
  output [7:0] out_cat_u,
  output [7:0] out_cat_s
);
  wire [3:0] _GEN_2 = uina4 % uinb4;
  wire [3:0] _GEN_3 = $signed(sina4) % $signed(sinb4);
  wire [18:0] _GEN_0 = {{15'd0}, uina4};
  wire [18:0] _GEN_1 = {{15{sina4[3]}},sina4};
  assign out_add_u = uina4 + uinb4;
  assign out_add_s = $signed(sina4) + $signed(sinb4);
  assign out_sub_u = uina4 - uinb4;
  assign out_sub_s = $signed(sina4) - $signed(sinb4);
  assign out_mul_u = uina4 * uinb4;
  assign out_mul_s = $signed(sina4) * $signed(sinb4);
  assign out_div_u = uina4 / uinb4;
  assign out_div_s = $signed(sina4) / $signed(sinb4);
  assign out_rem_u = _GEN_2[3:0];
  assign out_rem_s = _GEN_3[3:0];
  assign out_lt_u = uina4 < uinb4;
  assign out_lt_s = $signed(sina4) < $signed(sinb4);
  assign out_leq_u = uina4 <= uinb4;
  assign out_leq_s = $signed(sina4) <= $signed(sinb4);
  assign out_gt_u = uina4 > uinb4;
  assign out_gt_s = $signed(sina4) > $signed(sinb4);
  assign out_geq_u = uina4 >= uinb4;
  assign out_geq_s = $signed(sina4) >= $signed(sinb4);
  assign out_eq_u = uina4 == uinb4;
  assign out_eq_s = $signed(sina4) == $signed(sinb4);
  assign out_neq_u = uina4 != uinb4;
  assign out_neq_s = $signed(sina4) != $signed(sinb4);
  assign out_dshl_u = _GEN_0 << uinb4;
  assign out_dshl_s = $signed(_GEN_1) << uinb4;
  assign out_dshr_u = uina4 >> uinb4;
  assign out_dshr_s = $signed(sina4) >>> uinb4;
  assign out_and_u = uina4 & uinb4;
  assign out_and_s = $signed(sina4) & $signed(sinb4);
  assign out_or_u = uina4 | uinb4;
  assign out_or_s = $signed(sina4) | $signed(sinb4);
  assign out_xor_u = uina4 ^ uinb4;
  assign out_xor_s = $signed(sina4) ^ $signed(sinb4);
  assign out_cat_u = {uina4,uinb4};
  assign out_cat_s = {sina4,sinb4};
endmodule
