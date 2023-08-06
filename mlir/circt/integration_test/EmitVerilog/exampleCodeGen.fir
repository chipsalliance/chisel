; REQUIRES: verilator
; RUN: firtool -verilog %s > %t.sv
; RUN: circt-rtl-sim.py %t.sv 2>&1 | tee %t.out
; RUN: grep PASS %t.out

circuit top :
  module LFSR :
    input clock : Clock
    input reset : UInt<1>
    output io_out : UInt<6>
    output io_out3 : UInt<3>

    reg D0123456 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), D0123456) @[main.scala 16:25]
    node _T = bits(D0123456, 4, 0) @[main.scala 20:30]
    node _T_1 = bits(D0123456, 5, 5) @[main.scala 20:47]
    node _T_2 = cat(_T, _T_1) @[main.scala 20:36]
    node _T_3 = bits(D0123456, 4, 4) @[main.scala 20:59]
    node nxt_D0123456 = xor(_T_2, _T_3) @[main.scala 20:50]
    node _T_4 = bits(D0123456, 1, 1) @[main.scala 25:22]
    node _T_5 = bits(D0123456, 3, 3) @[main.scala 25:37]
    node _T_6 = cat(_T_4, _T_5) @[main.scala 25:26]
    node _T_7 = bits(D0123456, 5, 5) @[main.scala 25:52]
    node _T_8 = cat(_T_6, _T_7) @[main.scala 25:41]
    io_out <= D0123456 @[main.scala 24:10]
    io_out3 <= _T_8 @[main.scala 25:11]
    D0123456 <= mux(reset, UInt<6>("h1"), nxt_D0123456) @[main.scala 16:25 main.scala 16:25 main.scala 22:12]

  module top :
    input clk : Clock
    input rst : UInt<1>

    inst iLFSR of LFSR @[main.scala 31:21]
    node clock = clk
    node reset = rst
    reg i : UInt<8>, clock with :
      reset => (UInt<1>("h0"), i) @[main.scala 32:18]
    reg golden_0 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_0) @[main.scala 33:23]
    reg golden_1 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_1) @[main.scala 33:23]
    reg golden_2 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_2) @[main.scala 33:23]
    reg golden_3 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_3) @[main.scala 33:23]
    reg golden_4 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_4) @[main.scala 33:23]
    reg golden_5 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_5) @[main.scala 33:23]
    reg golden_6 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_6) @[main.scala 33:23]
    reg golden_7 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_7) @[main.scala 33:23]
    reg golden_8 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_8) @[main.scala 33:23]
    reg golden_9 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_9) @[main.scala 33:23]
    reg golden_10 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_10) @[main.scala 33:23]
    reg golden_11 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_11) @[main.scala 33:23]
    reg golden_12 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_12) @[main.scala 33:23]
    reg golden_13 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_13) @[main.scala 33:23]
    reg golden_14 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_14) @[main.scala 33:23]
    reg golden_15 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_15) @[main.scala 33:23]
    reg golden_16 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_16) @[main.scala 33:23]
    reg golden_17 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_17) @[main.scala 33:23]
    reg golden_18 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_18) @[main.scala 33:23]
    reg golden_19 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_19) @[main.scala 33:23]
    reg golden_20 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_20) @[main.scala 33:23]
    reg golden_21 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_21) @[main.scala 33:23]
    reg golden_22 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_22) @[main.scala 33:23]
    reg golden_23 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_23) @[main.scala 33:23]
    reg golden_24 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_24) @[main.scala 33:23]
    reg golden_25 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_25) @[main.scala 33:23]
    reg golden_26 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_26) @[main.scala 33:23]
    reg golden_27 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_27) @[main.scala 33:23]
    reg golden_28 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_28) @[main.scala 33:23]
    reg golden_29 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_29) @[main.scala 33:23]
    reg golden_30 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_30) @[main.scala 33:23]
    reg golden_31 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_31) @[main.scala 33:23]
    reg golden_32 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_32) @[main.scala 33:23]
    reg golden_33 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_33) @[main.scala 33:23]
    reg golden_34 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_34) @[main.scala 33:23]
    reg golden_35 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_35) @[main.scala 33:23]
    reg golden_36 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_36) @[main.scala 33:23]
    reg golden_37 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_37) @[main.scala 33:23]
    reg golden_38 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_38) @[main.scala 33:23]
    reg golden_39 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_39) @[main.scala 33:23]
    reg golden_40 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_40) @[main.scala 33:23]
    reg golden_41 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_41) @[main.scala 33:23]
    reg golden_42 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_42) @[main.scala 33:23]
    reg golden_43 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_43) @[main.scala 33:23]
    reg golden_44 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_44) @[main.scala 33:23]
    reg golden_45 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_45) @[main.scala 33:23]
    reg golden_46 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_46) @[main.scala 33:23]
    reg golden_47 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_47) @[main.scala 33:23]
    reg golden_48 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_48) @[main.scala 33:23]
    reg golden_49 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_49) @[main.scala 33:23]
    reg golden_50 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_50) @[main.scala 33:23]
    reg golden_51 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_51) @[main.scala 33:23]
    reg golden_52 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_52) @[main.scala 33:23]
    reg golden_53 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_53) @[main.scala 33:23]
    reg golden_54 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_54) @[main.scala 33:23]
    reg golden_55 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_55) @[main.scala 33:23]
    reg golden_56 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_56) @[main.scala 33:23]
    reg golden_57 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_57) @[main.scala 33:23]
    reg golden_58 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_58) @[main.scala 33:23]
    reg golden_59 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_59) @[main.scala 33:23]
    reg golden_60 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_60) @[main.scala 33:23]
    reg golden_61 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_61) @[main.scala 33:23]
    reg golden_62 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_62) @[main.scala 33:23]
    reg golden_63 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_63) @[main.scala 33:23]
    reg golden_64 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_64) @[main.scala 33:23]
    reg golden_65 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_65) @[main.scala 33:23]
    reg golden_66 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_66) @[main.scala 33:23]
    reg golden_67 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_67) @[main.scala 33:23]
    reg golden_68 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_68) @[main.scala 33:23]
    reg golden_69 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_69) @[main.scala 33:23]
    reg golden_70 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_70) @[main.scala 33:23]
    reg golden_71 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_71) @[main.scala 33:23]
    reg golden_72 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_72) @[main.scala 33:23]
    reg golden_73 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_73) @[main.scala 33:23]
    reg golden_74 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_74) @[main.scala 33:23]
    reg golden_75 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_75) @[main.scala 33:23]
    reg golden_76 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_76) @[main.scala 33:23]
    reg golden_77 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_77) @[main.scala 33:23]
    reg golden_78 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_78) @[main.scala 33:23]
    reg golden_79 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_79) @[main.scala 33:23]
    reg golden_80 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_80) @[main.scala 33:23]
    reg golden_81 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_81) @[main.scala 33:23]
    reg golden_82 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_82) @[main.scala 33:23]
    reg golden_83 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_83) @[main.scala 33:23]
    reg golden_84 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_84) @[main.scala 33:23]
    reg golden_85 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_85) @[main.scala 33:23]
    reg golden_86 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_86) @[main.scala 33:23]
    reg golden_87 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_87) @[main.scala 33:23]
    reg golden_88 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_88) @[main.scala 33:23]
    reg golden_89 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_89) @[main.scala 33:23]
    reg golden_90 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_90) @[main.scala 33:23]
    reg golden_91 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_91) @[main.scala 33:23]
    reg golden_92 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_92) @[main.scala 33:23]
    reg golden_93 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_93) @[main.scala 33:23]
    reg golden_94 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_94) @[main.scala 33:23]
    reg golden_95 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_95) @[main.scala 33:23]
    reg golden_96 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_96) @[main.scala 33:23]
    reg golden_97 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_97) @[main.scala 33:23]
    reg golden_98 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_98) @[main.scala 33:23]
    reg golden_99 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_99) @[main.scala 33:23]
    reg golden_100 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_100) @[main.scala 33:23]
    reg golden_101 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_101) @[main.scala 33:23]
    reg golden_102 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_102) @[main.scala 33:23]
    reg golden_103 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_103) @[main.scala 33:23]
    reg golden_104 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_104) @[main.scala 33:23]
    reg golden_105 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_105) @[main.scala 33:23]
    reg golden_106 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_106) @[main.scala 33:23]
    reg golden_107 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_107) @[main.scala 33:23]
    reg golden_108 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_108) @[main.scala 33:23]
    reg golden_109 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_109) @[main.scala 33:23]
    reg golden_110 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_110) @[main.scala 33:23]
    reg golden_111 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_111) @[main.scala 33:23]
    reg golden_112 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_112) @[main.scala 33:23]
    reg golden_113 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_113) @[main.scala 33:23]
    reg golden_114 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_114) @[main.scala 33:23]
    reg golden_115 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_115) @[main.scala 33:23]
    reg golden_116 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_116) @[main.scala 33:23]
    reg golden_117 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_117) @[main.scala 33:23]
    reg golden_118 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_118) @[main.scala 33:23]
    reg golden_119 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_119) @[main.scala 33:23]
    reg golden_120 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_120) @[main.scala 33:23]
    reg golden_121 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_121) @[main.scala 33:23]
    reg golden_122 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_122) @[main.scala 33:23]
    reg golden_123 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_123) @[main.scala 33:23]
    reg golden_124 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_124) @[main.scala 33:23]
    reg golden_125 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_125) @[main.scala 33:23]
    reg golden_126 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_126) @[main.scala 33:23]
    reg golden_127 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_127) @[main.scala 33:23]
    reg golden_128 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_128) @[main.scala 33:23]
    reg golden_129 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_129) @[main.scala 33:23]
    reg golden_130 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_130) @[main.scala 33:23]
    reg golden_131 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_131) @[main.scala 33:23]
    reg golden_132 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_132) @[main.scala 33:23]
    reg golden_133 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_133) @[main.scala 33:23]
    reg golden_134 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_134) @[main.scala 33:23]
    reg golden_135 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_135) @[main.scala 33:23]
    reg golden_136 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_136) @[main.scala 33:23]
    reg golden_137 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_137) @[main.scala 33:23]
    reg golden_138 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_138) @[main.scala 33:23]
    reg golden_139 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_139) @[main.scala 33:23]
    reg golden_140 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_140) @[main.scala 33:23]
    reg golden_141 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_141) @[main.scala 33:23]
    reg golden_142 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_142) @[main.scala 33:23]
    reg golden_143 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_143) @[main.scala 33:23]
    reg golden_144 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_144) @[main.scala 33:23]
    reg golden_145 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_145) @[main.scala 33:23]
    reg golden_146 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_146) @[main.scala 33:23]
    reg golden_147 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_147) @[main.scala 33:23]
    reg golden_148 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_148) @[main.scala 33:23]
    reg golden_149 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_149) @[main.scala 33:23]
    reg golden_150 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_150) @[main.scala 33:23]
    reg golden_151 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_151) @[main.scala 33:23]
    reg golden_152 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_152) @[main.scala 33:23]
    reg golden_153 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_153) @[main.scala 33:23]
    reg golden_154 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_154) @[main.scala 33:23]
    reg golden_155 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_155) @[main.scala 33:23]
    reg golden_156 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_156) @[main.scala 33:23]
    reg golden_157 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_157) @[main.scala 33:23]
    reg golden_158 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_158) @[main.scala 33:23]
    reg golden_159 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_159) @[main.scala 33:23]
    reg golden_160 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_160) @[main.scala 33:23]
    reg golden_161 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_161) @[main.scala 33:23]
    reg golden_162 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_162) @[main.scala 33:23]
    reg golden_163 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_163) @[main.scala 33:23]
    reg golden_164 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_164) @[main.scala 33:23]
    reg golden_165 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_165) @[main.scala 33:23]
    reg golden_166 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_166) @[main.scala 33:23]
    reg golden_167 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_167) @[main.scala 33:23]
    reg golden_168 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_168) @[main.scala 33:23]
    reg golden_169 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_169) @[main.scala 33:23]
    reg golden_170 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_170) @[main.scala 33:23]
    reg golden_171 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_171) @[main.scala 33:23]
    reg golden_172 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_172) @[main.scala 33:23]
    reg golden_173 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_173) @[main.scala 33:23]
    reg golden_174 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_174) @[main.scala 33:23]
    reg golden_175 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_175) @[main.scala 33:23]
    reg golden_176 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_176) @[main.scala 33:23]
    reg golden_177 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_177) @[main.scala 33:23]
    reg golden_178 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_178) @[main.scala 33:23]
    reg golden_179 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_179) @[main.scala 33:23]
    reg golden_180 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_180) @[main.scala 33:23]
    reg golden_181 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_181) @[main.scala 33:23]
    reg golden_182 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_182) @[main.scala 33:23]
    reg golden_183 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_183) @[main.scala 33:23]
    reg golden_184 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_184) @[main.scala 33:23]
    reg golden_185 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_185) @[main.scala 33:23]
    reg golden_186 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_186) @[main.scala 33:23]
    reg golden_187 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_187) @[main.scala 33:23]
    reg golden_188 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_188) @[main.scala 33:23]
    reg golden_189 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_189) @[main.scala 33:23]
    reg golden_190 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_190) @[main.scala 33:23]
    reg golden_191 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_191) @[main.scala 33:23]
    reg golden_192 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_192) @[main.scala 33:23]
    reg golden_193 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_193) @[main.scala 33:23]
    reg golden_194 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_194) @[main.scala 33:23]
    reg golden_195 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_195) @[main.scala 33:23]
    reg golden_196 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_196) @[main.scala 33:23]
    reg golden_197 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_197) @[main.scala 33:23]
    reg golden_198 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_198) @[main.scala 33:23]
    reg golden_199 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_199) @[main.scala 33:23]
    reg golden_200 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_200) @[main.scala 33:23]
    reg golden_201 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_201) @[main.scala 33:23]
    reg golden_202 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_202) @[main.scala 33:23]
    reg golden_203 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_203) @[main.scala 33:23]
    reg golden_204 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_204) @[main.scala 33:23]
    reg golden_205 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_205) @[main.scala 33:23]
    reg golden_206 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_206) @[main.scala 33:23]
    reg golden_207 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_207) @[main.scala 33:23]
    reg golden_208 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_208) @[main.scala 33:23]
    reg golden_209 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_209) @[main.scala 33:23]
    reg golden_210 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_210) @[main.scala 33:23]
    reg golden_211 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_211) @[main.scala 33:23]
    reg golden_212 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_212) @[main.scala 33:23]
    reg golden_213 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_213) @[main.scala 33:23]
    reg golden_214 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_214) @[main.scala 33:23]
    reg golden_215 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_215) @[main.scala 33:23]
    reg golden_216 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_216) @[main.scala 33:23]
    reg golden_217 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_217) @[main.scala 33:23]
    reg golden_218 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_218) @[main.scala 33:23]
    reg golden_219 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_219) @[main.scala 33:23]
    reg golden_220 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_220) @[main.scala 33:23]
    reg golden_221 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_221) @[main.scala 33:23]
    reg golden_222 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_222) @[main.scala 33:23]
    reg golden_223 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_223) @[main.scala 33:23]
    reg golden_224 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_224) @[main.scala 33:23]
    reg golden_225 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_225) @[main.scala 33:23]
    reg golden_226 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_226) @[main.scala 33:23]
    reg golden_227 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_227) @[main.scala 33:23]
    reg golden_228 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_228) @[main.scala 33:23]
    reg golden_229 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_229) @[main.scala 33:23]
    reg golden_230 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_230) @[main.scala 33:23]
    reg golden_231 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_231) @[main.scala 33:23]
    reg golden_232 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_232) @[main.scala 33:23]
    reg golden_233 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_233) @[main.scala 33:23]
    reg golden_234 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_234) @[main.scala 33:23]
    reg golden_235 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_235) @[main.scala 33:23]
    reg golden_236 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_236) @[main.scala 33:23]
    reg golden_237 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_237) @[main.scala 33:23]
    reg golden_238 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_238) @[main.scala 33:23]
    reg golden_239 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_239) @[main.scala 33:23]
    reg golden_240 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_240) @[main.scala 33:23]
    reg golden_241 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_241) @[main.scala 33:23]
    reg golden_242 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_242) @[main.scala 33:23]
    reg golden_243 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_243) @[main.scala 33:23]
    reg golden_244 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_244) @[main.scala 33:23]
    reg golden_245 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_245) @[main.scala 33:23]
    reg golden_246 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_246) @[main.scala 33:23]
    reg golden_247 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_247) @[main.scala 33:23]
    reg golden_248 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_248) @[main.scala 33:23]
    reg golden_249 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_249) @[main.scala 33:23]
    reg golden_250 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_250) @[main.scala 33:23]
    reg golden_251 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_251) @[main.scala 33:23]
    reg golden_252 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_252) @[main.scala 33:23]
    reg golden_253 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_253) @[main.scala 33:23]
    reg golden_254 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_254) @[main.scala 33:23]
    reg golden_255 : UInt<6>, clock with :
      reset => (UInt<1>("h0"), golden_255) @[main.scala 33:23]
    reg pass : UInt<1>, clock with :
      reset => (UInt<1>("h0"), pass) @[main.scala 66:21]
    node _T = eq(i, UInt<8>("hff")) @[main.scala 67:11]
    node _T_1 = eq(pass, UInt<1>("h1")) @[main.scala 68:15]
    node _T_2 = bits(reset, 0, 0) @[main.scala 69:13]
    node _T_3 = eq(_T_2, UInt<1>("h0")) @[main.scala 69:13]
    node _T_4 = eq(pass, UInt<1>("h0")) @[main.scala 71:15]
    node _T_5 = bits(reset, 0, 0) @[main.scala 72:13]
    node _T_6 = eq(_T_5, UInt<1>("h0")) @[main.scala 72:13]
    node _T_7 = bits(reset, 0, 0) @[main.scala 74:9]
    node _T_8 = eq(_T_7, UInt<1>("h0")) @[main.scala 74:9]
    node _T_9 = add(i, UInt<1>("h1")) @[main.scala 76:10]
    node _T_10 = tail(_T_9, 1) @[main.scala 76:10]
    node _GEN_0 = golden_0 @[main.scala 77:32 main.scala 77:32]
    node _GEN_1 = mux(eq(UInt<1>("h1"), i), golden_1, _GEN_0) @[main.scala 77:32 main.scala 77:32]
    node _GEN_2 = mux(eq(UInt<2>("h2"), i), golden_2, _GEN_1) @[main.scala 77:32 main.scala 77:32]
    node _GEN_3 = mux(eq(UInt<2>("h3"), i), golden_3, _GEN_2) @[main.scala 77:32 main.scala 77:32]
    node _GEN_4 = mux(eq(UInt<3>("h4"), i), golden_4, _GEN_3) @[main.scala 77:32 main.scala 77:32]
    node _GEN_5 = mux(eq(UInt<3>("h5"), i), golden_5, _GEN_4) @[main.scala 77:32 main.scala 77:32]
    node _GEN_6 = mux(eq(UInt<3>("h6"), i), golden_6, _GEN_5) @[main.scala 77:32 main.scala 77:32]
    node _GEN_7 = mux(eq(UInt<3>("h7"), i), golden_7, _GEN_6) @[main.scala 77:32 main.scala 77:32]
    node _GEN_8 = mux(eq(UInt<4>("h8"), i), golden_8, _GEN_7) @[main.scala 77:32 main.scala 77:32]
    node _GEN_9 = mux(eq(UInt<4>("h9"), i), golden_9, _GEN_8) @[main.scala 77:32 main.scala 77:32]
    node _GEN_10 = mux(eq(UInt<4>("ha"), i), golden_10, _GEN_9) @[main.scala 77:32 main.scala 77:32]
    node _GEN_11 = mux(eq(UInt<4>("hb"), i), golden_11, _GEN_10) @[main.scala 77:32 main.scala 77:32]
    node _GEN_12 = mux(eq(UInt<4>("hc"), i), golden_12, _GEN_11) @[main.scala 77:32 main.scala 77:32]
    node _GEN_13 = mux(eq(UInt<4>("hd"), i), golden_13, _GEN_12) @[main.scala 77:32 main.scala 77:32]
    node _GEN_14 = mux(eq(UInt<4>("he"), i), golden_14, _GEN_13) @[main.scala 77:32 main.scala 77:32]
    node _GEN_15 = mux(eq(UInt<4>("hf"), i), golden_15, _GEN_14) @[main.scala 77:32 main.scala 77:32]
    node _GEN_16 = mux(eq(UInt<5>("h10"), i), golden_16, _GEN_15) @[main.scala 77:32 main.scala 77:32]
    node _GEN_17 = mux(eq(UInt<5>("h11"), i), golden_17, _GEN_16) @[main.scala 77:32 main.scala 77:32]
    node _GEN_18 = mux(eq(UInt<5>("h12"), i), golden_18, _GEN_17) @[main.scala 77:32 main.scala 77:32]
    node _GEN_19 = mux(eq(UInt<5>("h13"), i), golden_19, _GEN_18) @[main.scala 77:32 main.scala 77:32]
    node _GEN_20 = mux(eq(UInt<5>("h14"), i), golden_20, _GEN_19) @[main.scala 77:32 main.scala 77:32]
    node _GEN_21 = mux(eq(UInt<5>("h15"), i), golden_21, _GEN_20) @[main.scala 77:32 main.scala 77:32]
    node _GEN_22 = mux(eq(UInt<5>("h16"), i), golden_22, _GEN_21) @[main.scala 77:32 main.scala 77:32]
    node _GEN_23 = mux(eq(UInt<5>("h17"), i), golden_23, _GEN_22) @[main.scala 77:32 main.scala 77:32]
    node _GEN_24 = mux(eq(UInt<5>("h18"), i), golden_24, _GEN_23) @[main.scala 77:32 main.scala 77:32]
    node _GEN_25 = mux(eq(UInt<5>("h19"), i), golden_25, _GEN_24) @[main.scala 77:32 main.scala 77:32]
    node _GEN_26 = mux(eq(UInt<5>("h1a"), i), golden_26, _GEN_25) @[main.scala 77:32 main.scala 77:32]
    node _GEN_27 = mux(eq(UInt<5>("h1b"), i), golden_27, _GEN_26) @[main.scala 77:32 main.scala 77:32]
    node _GEN_28 = mux(eq(UInt<5>("h1c"), i), golden_28, _GEN_27) @[main.scala 77:32 main.scala 77:32]
    node _GEN_29 = mux(eq(UInt<5>("h1d"), i), golden_29, _GEN_28) @[main.scala 77:32 main.scala 77:32]
    node _GEN_30 = mux(eq(UInt<5>("h1e"), i), golden_30, _GEN_29) @[main.scala 77:32 main.scala 77:32]
    node _GEN_31 = mux(eq(UInt<5>("h1f"), i), golden_31, _GEN_30) @[main.scala 77:32 main.scala 77:32]
    node _GEN_32 = mux(eq(UInt<6>("h20"), i), golden_32, _GEN_31) @[main.scala 77:32 main.scala 77:32]
    node _GEN_33 = mux(eq(UInt<6>("h21"), i), golden_33, _GEN_32) @[main.scala 77:32 main.scala 77:32]
    node _GEN_34 = mux(eq(UInt<6>("h22"), i), golden_34, _GEN_33) @[main.scala 77:32 main.scala 77:32]
    node _GEN_35 = mux(eq(UInt<6>("h23"), i), golden_35, _GEN_34) @[main.scala 77:32 main.scala 77:32]
    node _GEN_36 = mux(eq(UInt<6>("h24"), i), golden_36, _GEN_35) @[main.scala 77:32 main.scala 77:32]
    node _GEN_37 = mux(eq(UInt<6>("h25"), i), golden_37, _GEN_36) @[main.scala 77:32 main.scala 77:32]
    node _GEN_38 = mux(eq(UInt<6>("h26"), i), golden_38, _GEN_37) @[main.scala 77:32 main.scala 77:32]
    node _GEN_39 = mux(eq(UInt<6>("h27"), i), golden_39, _GEN_38) @[main.scala 77:32 main.scala 77:32]
    node _GEN_40 = mux(eq(UInt<6>("h28"), i), golden_40, _GEN_39) @[main.scala 77:32 main.scala 77:32]
    node _GEN_41 = mux(eq(UInt<6>("h29"), i), golden_41, _GEN_40) @[main.scala 77:32 main.scala 77:32]
    node _GEN_42 = mux(eq(UInt<6>("h2a"), i), golden_42, _GEN_41) @[main.scala 77:32 main.scala 77:32]
    node _GEN_43 = mux(eq(UInt<6>("h2b"), i), golden_43, _GEN_42) @[main.scala 77:32 main.scala 77:32]
    node _GEN_44 = mux(eq(UInt<6>("h2c"), i), golden_44, _GEN_43) @[main.scala 77:32 main.scala 77:32]
    node _GEN_45 = mux(eq(UInt<6>("h2d"), i), golden_45, _GEN_44) @[main.scala 77:32 main.scala 77:32]
    node _GEN_46 = mux(eq(UInt<6>("h2e"), i), golden_46, _GEN_45) @[main.scala 77:32 main.scala 77:32]
    node _GEN_47 = mux(eq(UInt<6>("h2f"), i), golden_47, _GEN_46) @[main.scala 77:32 main.scala 77:32]
    node _GEN_48 = mux(eq(UInt<6>("h30"), i), golden_48, _GEN_47) @[main.scala 77:32 main.scala 77:32]
    node _GEN_49 = mux(eq(UInt<6>("h31"), i), golden_49, _GEN_48) @[main.scala 77:32 main.scala 77:32]
    node _GEN_50 = mux(eq(UInt<6>("h32"), i), golden_50, _GEN_49) @[main.scala 77:32 main.scala 77:32]
    node _GEN_51 = mux(eq(UInt<6>("h33"), i), golden_51, _GEN_50) @[main.scala 77:32 main.scala 77:32]
    node _GEN_52 = mux(eq(UInt<6>("h34"), i), golden_52, _GEN_51) @[main.scala 77:32 main.scala 77:32]
    node _GEN_53 = mux(eq(UInt<6>("h35"), i), golden_53, _GEN_52) @[main.scala 77:32 main.scala 77:32]
    node _GEN_54 = mux(eq(UInt<6>("h36"), i), golden_54, _GEN_53) @[main.scala 77:32 main.scala 77:32]
    node _GEN_55 = mux(eq(UInt<6>("h37"), i), golden_55, _GEN_54) @[main.scala 77:32 main.scala 77:32]
    node _GEN_56 = mux(eq(UInt<6>("h38"), i), golden_56, _GEN_55) @[main.scala 77:32 main.scala 77:32]
    node _GEN_57 = mux(eq(UInt<6>("h39"), i), golden_57, _GEN_56) @[main.scala 77:32 main.scala 77:32]
    node _GEN_58 = mux(eq(UInt<6>("h3a"), i), golden_58, _GEN_57) @[main.scala 77:32 main.scala 77:32]
    node _GEN_59 = mux(eq(UInt<6>("h3b"), i), golden_59, _GEN_58) @[main.scala 77:32 main.scala 77:32]
    node _GEN_60 = mux(eq(UInt<6>("h3c"), i), golden_60, _GEN_59) @[main.scala 77:32 main.scala 77:32]
    node _GEN_61 = mux(eq(UInt<6>("h3d"), i), golden_61, _GEN_60) @[main.scala 77:32 main.scala 77:32]
    node _GEN_62 = mux(eq(UInt<6>("h3e"), i), golden_62, _GEN_61) @[main.scala 77:32 main.scala 77:32]
    node _GEN_63 = mux(eq(UInt<6>("h3f"), i), golden_63, _GEN_62) @[main.scala 77:32 main.scala 77:32]
    node _GEN_64 = mux(eq(UInt<7>("h40"), i), golden_64, _GEN_63) @[main.scala 77:32 main.scala 77:32]
    node _GEN_65 = mux(eq(UInt<7>("h41"), i), golden_65, _GEN_64) @[main.scala 77:32 main.scala 77:32]
    node _GEN_66 = mux(eq(UInt<7>("h42"), i), golden_66, _GEN_65) @[main.scala 77:32 main.scala 77:32]
    node _GEN_67 = mux(eq(UInt<7>("h43"), i), golden_67, _GEN_66) @[main.scala 77:32 main.scala 77:32]
    node _GEN_68 = mux(eq(UInt<7>("h44"), i), golden_68, _GEN_67) @[main.scala 77:32 main.scala 77:32]
    node _GEN_69 = mux(eq(UInt<7>("h45"), i), golden_69, _GEN_68) @[main.scala 77:32 main.scala 77:32]
    node _GEN_70 = mux(eq(UInt<7>("h46"), i), golden_70, _GEN_69) @[main.scala 77:32 main.scala 77:32]
    node _GEN_71 = mux(eq(UInt<7>("h47"), i), golden_71, _GEN_70) @[main.scala 77:32 main.scala 77:32]
    node _GEN_72 = mux(eq(UInt<7>("h48"), i), golden_72, _GEN_71) @[main.scala 77:32 main.scala 77:32]
    node _GEN_73 = mux(eq(UInt<7>("h49"), i), golden_73, _GEN_72) @[main.scala 77:32 main.scala 77:32]
    node _GEN_74 = mux(eq(UInt<7>("h4a"), i), golden_74, _GEN_73) @[main.scala 77:32 main.scala 77:32]
    node _GEN_75 = mux(eq(UInt<7>("h4b"), i), golden_75, _GEN_74) @[main.scala 77:32 main.scala 77:32]
    node _GEN_76 = mux(eq(UInt<7>("h4c"), i), golden_76, _GEN_75) @[main.scala 77:32 main.scala 77:32]
    node _GEN_77 = mux(eq(UInt<7>("h4d"), i), golden_77, _GEN_76) @[main.scala 77:32 main.scala 77:32]
    node _GEN_78 = mux(eq(UInt<7>("h4e"), i), golden_78, _GEN_77) @[main.scala 77:32 main.scala 77:32]
    node _GEN_79 = mux(eq(UInt<7>("h4f"), i), golden_79, _GEN_78) @[main.scala 77:32 main.scala 77:32]
    node _GEN_80 = mux(eq(UInt<7>("h50"), i), golden_80, _GEN_79) @[main.scala 77:32 main.scala 77:32]
    node _GEN_81 = mux(eq(UInt<7>("h51"), i), golden_81, _GEN_80) @[main.scala 77:32 main.scala 77:32]
    node _GEN_82 = mux(eq(UInt<7>("h52"), i), golden_82, _GEN_81) @[main.scala 77:32 main.scala 77:32]
    node _GEN_83 = mux(eq(UInt<7>("h53"), i), golden_83, _GEN_82) @[main.scala 77:32 main.scala 77:32]
    node _GEN_84 = mux(eq(UInt<7>("h54"), i), golden_84, _GEN_83) @[main.scala 77:32 main.scala 77:32]
    node _GEN_85 = mux(eq(UInt<7>("h55"), i), golden_85, _GEN_84) @[main.scala 77:32 main.scala 77:32]
    node _GEN_86 = mux(eq(UInt<7>("h56"), i), golden_86, _GEN_85) @[main.scala 77:32 main.scala 77:32]
    node _GEN_87 = mux(eq(UInt<7>("h57"), i), golden_87, _GEN_86) @[main.scala 77:32 main.scala 77:32]
    node _GEN_88 = mux(eq(UInt<7>("h58"), i), golden_88, _GEN_87) @[main.scala 77:32 main.scala 77:32]
    node _GEN_89 = mux(eq(UInt<7>("h59"), i), golden_89, _GEN_88) @[main.scala 77:32 main.scala 77:32]
    node _GEN_90 = mux(eq(UInt<7>("h5a"), i), golden_90, _GEN_89) @[main.scala 77:32 main.scala 77:32]
    node _GEN_91 = mux(eq(UInt<7>("h5b"), i), golden_91, _GEN_90) @[main.scala 77:32 main.scala 77:32]
    node _GEN_92 = mux(eq(UInt<7>("h5c"), i), golden_92, _GEN_91) @[main.scala 77:32 main.scala 77:32]
    node _GEN_93 = mux(eq(UInt<7>("h5d"), i), golden_93, _GEN_92) @[main.scala 77:32 main.scala 77:32]
    node _GEN_94 = mux(eq(UInt<7>("h5e"), i), golden_94, _GEN_93) @[main.scala 77:32 main.scala 77:32]
    node _GEN_95 = mux(eq(UInt<7>("h5f"), i), golden_95, _GEN_94) @[main.scala 77:32 main.scala 77:32]
    node _GEN_96 = mux(eq(UInt<7>("h60"), i), golden_96, _GEN_95) @[main.scala 77:32 main.scala 77:32]
    node _GEN_97 = mux(eq(UInt<7>("h61"), i), golden_97, _GEN_96) @[main.scala 77:32 main.scala 77:32]
    node _GEN_98 = mux(eq(UInt<7>("h62"), i), golden_98, _GEN_97) @[main.scala 77:32 main.scala 77:32]
    node _GEN_99 = mux(eq(UInt<7>("h63"), i), golden_99, _GEN_98) @[main.scala 77:32 main.scala 77:32]
    node _GEN_100 = mux(eq(UInt<7>("h64"), i), golden_100, _GEN_99) @[main.scala 77:32 main.scala 77:32]
    node _GEN_101 = mux(eq(UInt<7>("h65"), i), golden_101, _GEN_100) @[main.scala 77:32 main.scala 77:32]
    node _GEN_102 = mux(eq(UInt<7>("h66"), i), golden_102, _GEN_101) @[main.scala 77:32 main.scala 77:32]
    node _GEN_103 = mux(eq(UInt<7>("h67"), i), golden_103, _GEN_102) @[main.scala 77:32 main.scala 77:32]
    node _GEN_104 = mux(eq(UInt<7>("h68"), i), golden_104, _GEN_103) @[main.scala 77:32 main.scala 77:32]
    node _GEN_105 = mux(eq(UInt<7>("h69"), i), golden_105, _GEN_104) @[main.scala 77:32 main.scala 77:32]
    node _GEN_106 = mux(eq(UInt<7>("h6a"), i), golden_106, _GEN_105) @[main.scala 77:32 main.scala 77:32]
    node _GEN_107 = mux(eq(UInt<7>("h6b"), i), golden_107, _GEN_106) @[main.scala 77:32 main.scala 77:32]
    node _GEN_108 = mux(eq(UInt<7>("h6c"), i), golden_108, _GEN_107) @[main.scala 77:32 main.scala 77:32]
    node _GEN_109 = mux(eq(UInt<7>("h6d"), i), golden_109, _GEN_108) @[main.scala 77:32 main.scala 77:32]
    node _GEN_110 = mux(eq(UInt<7>("h6e"), i), golden_110, _GEN_109) @[main.scala 77:32 main.scala 77:32]
    node _GEN_111 = mux(eq(UInt<7>("h6f"), i), golden_111, _GEN_110) @[main.scala 77:32 main.scala 77:32]
    node _GEN_112 = mux(eq(UInt<7>("h70"), i), golden_112, _GEN_111) @[main.scala 77:32 main.scala 77:32]
    node _GEN_113 = mux(eq(UInt<7>("h71"), i), golden_113, _GEN_112) @[main.scala 77:32 main.scala 77:32]
    node _GEN_114 = mux(eq(UInt<7>("h72"), i), golden_114, _GEN_113) @[main.scala 77:32 main.scala 77:32]
    node _GEN_115 = mux(eq(UInt<7>("h73"), i), golden_115, _GEN_114) @[main.scala 77:32 main.scala 77:32]
    node _GEN_116 = mux(eq(UInt<7>("h74"), i), golden_116, _GEN_115) @[main.scala 77:32 main.scala 77:32]
    node _GEN_117 = mux(eq(UInt<7>("h75"), i), golden_117, _GEN_116) @[main.scala 77:32 main.scala 77:32]
    node _GEN_118 = mux(eq(UInt<7>("h76"), i), golden_118, _GEN_117) @[main.scala 77:32 main.scala 77:32]
    node _GEN_119 = mux(eq(UInt<7>("h77"), i), golden_119, _GEN_118) @[main.scala 77:32 main.scala 77:32]
    node _GEN_120 = mux(eq(UInt<7>("h78"), i), golden_120, _GEN_119) @[main.scala 77:32 main.scala 77:32]
    node _GEN_121 = mux(eq(UInt<7>("h79"), i), golden_121, _GEN_120) @[main.scala 77:32 main.scala 77:32]
    node _GEN_122 = mux(eq(UInt<7>("h7a"), i), golden_122, _GEN_121) @[main.scala 77:32 main.scala 77:32]
    node _GEN_123 = mux(eq(UInt<7>("h7b"), i), golden_123, _GEN_122) @[main.scala 77:32 main.scala 77:32]
    node _GEN_124 = mux(eq(UInt<7>("h7c"), i), golden_124, _GEN_123) @[main.scala 77:32 main.scala 77:32]
    node _GEN_125 = mux(eq(UInt<7>("h7d"), i), golden_125, _GEN_124) @[main.scala 77:32 main.scala 77:32]
    node _GEN_126 = mux(eq(UInt<7>("h7e"), i), golden_126, _GEN_125) @[main.scala 77:32 main.scala 77:32]
    node _GEN_127 = mux(eq(UInt<7>("h7f"), i), golden_127, _GEN_126) @[main.scala 77:32 main.scala 77:32]
    node _GEN_128 = mux(eq(UInt<8>("h80"), i), golden_128, _GEN_127) @[main.scala 77:32 main.scala 77:32]
    node _GEN_129 = mux(eq(UInt<8>("h81"), i), golden_129, _GEN_128) @[main.scala 77:32 main.scala 77:32]
    node _GEN_130 = mux(eq(UInt<8>("h82"), i), golden_130, _GEN_129) @[main.scala 77:32 main.scala 77:32]
    node _GEN_131 = mux(eq(UInt<8>("h83"), i), golden_131, _GEN_130) @[main.scala 77:32 main.scala 77:32]
    node _GEN_132 = mux(eq(UInt<8>("h84"), i), golden_132, _GEN_131) @[main.scala 77:32 main.scala 77:32]
    node _GEN_133 = mux(eq(UInt<8>("h85"), i), golden_133, _GEN_132) @[main.scala 77:32 main.scala 77:32]
    node _GEN_134 = mux(eq(UInt<8>("h86"), i), golden_134, _GEN_133) @[main.scala 77:32 main.scala 77:32]
    node _GEN_135 = mux(eq(UInt<8>("h87"), i), golden_135, _GEN_134) @[main.scala 77:32 main.scala 77:32]
    node _GEN_136 = mux(eq(UInt<8>("h88"), i), golden_136, _GEN_135) @[main.scala 77:32 main.scala 77:32]
    node _GEN_137 = mux(eq(UInt<8>("h89"), i), golden_137, _GEN_136) @[main.scala 77:32 main.scala 77:32]
    node _GEN_138 = mux(eq(UInt<8>("h8a"), i), golden_138, _GEN_137) @[main.scala 77:32 main.scala 77:32]
    node _GEN_139 = mux(eq(UInt<8>("h8b"), i), golden_139, _GEN_138) @[main.scala 77:32 main.scala 77:32]
    node _GEN_140 = mux(eq(UInt<8>("h8c"), i), golden_140, _GEN_139) @[main.scala 77:32 main.scala 77:32]
    node _GEN_141 = mux(eq(UInt<8>("h8d"), i), golden_141, _GEN_140) @[main.scala 77:32 main.scala 77:32]
    node _GEN_142 = mux(eq(UInt<8>("h8e"), i), golden_142, _GEN_141) @[main.scala 77:32 main.scala 77:32]
    node _GEN_143 = mux(eq(UInt<8>("h8f"), i), golden_143, _GEN_142) @[main.scala 77:32 main.scala 77:32]
    node _GEN_144 = mux(eq(UInt<8>("h90"), i), golden_144, _GEN_143) @[main.scala 77:32 main.scala 77:32]
    node _GEN_145 = mux(eq(UInt<8>("h91"), i), golden_145, _GEN_144) @[main.scala 77:32 main.scala 77:32]
    node _GEN_146 = mux(eq(UInt<8>("h92"), i), golden_146, _GEN_145) @[main.scala 77:32 main.scala 77:32]
    node _GEN_147 = mux(eq(UInt<8>("h93"), i), golden_147, _GEN_146) @[main.scala 77:32 main.scala 77:32]
    node _GEN_148 = mux(eq(UInt<8>("h94"), i), golden_148, _GEN_147) @[main.scala 77:32 main.scala 77:32]
    node _GEN_149 = mux(eq(UInt<8>("h95"), i), golden_149, _GEN_148) @[main.scala 77:32 main.scala 77:32]
    node _GEN_150 = mux(eq(UInt<8>("h96"), i), golden_150, _GEN_149) @[main.scala 77:32 main.scala 77:32]
    node _GEN_151 = mux(eq(UInt<8>("h97"), i), golden_151, _GEN_150) @[main.scala 77:32 main.scala 77:32]
    node _GEN_152 = mux(eq(UInt<8>("h98"), i), golden_152, _GEN_151) @[main.scala 77:32 main.scala 77:32]
    node _GEN_153 = mux(eq(UInt<8>("h99"), i), golden_153, _GEN_152) @[main.scala 77:32 main.scala 77:32]
    node _GEN_154 = mux(eq(UInt<8>("h9a"), i), golden_154, _GEN_153) @[main.scala 77:32 main.scala 77:32]
    node _GEN_155 = mux(eq(UInt<8>("h9b"), i), golden_155, _GEN_154) @[main.scala 77:32 main.scala 77:32]
    node _GEN_156 = mux(eq(UInt<8>("h9c"), i), golden_156, _GEN_155) @[main.scala 77:32 main.scala 77:32]
    node _GEN_157 = mux(eq(UInt<8>("h9d"), i), golden_157, _GEN_156) @[main.scala 77:32 main.scala 77:32]
    node _GEN_158 = mux(eq(UInt<8>("h9e"), i), golden_158, _GEN_157) @[main.scala 77:32 main.scala 77:32]
    node _GEN_159 = mux(eq(UInt<8>("h9f"), i), golden_159, _GEN_158) @[main.scala 77:32 main.scala 77:32]
    node _GEN_160 = mux(eq(UInt<8>("ha0"), i), golden_160, _GEN_159) @[main.scala 77:32 main.scala 77:32]
    node _GEN_161 = mux(eq(UInt<8>("ha1"), i), golden_161, _GEN_160) @[main.scala 77:32 main.scala 77:32]
    node _GEN_162 = mux(eq(UInt<8>("ha2"), i), golden_162, _GEN_161) @[main.scala 77:32 main.scala 77:32]
    node _GEN_163 = mux(eq(UInt<8>("ha3"), i), golden_163, _GEN_162) @[main.scala 77:32 main.scala 77:32]
    node _GEN_164 = mux(eq(UInt<8>("ha4"), i), golden_164, _GEN_163) @[main.scala 77:32 main.scala 77:32]
    node _GEN_165 = mux(eq(UInt<8>("ha5"), i), golden_165, _GEN_164) @[main.scala 77:32 main.scala 77:32]
    node _GEN_166 = mux(eq(UInt<8>("ha6"), i), golden_166, _GEN_165) @[main.scala 77:32 main.scala 77:32]
    node _GEN_167 = mux(eq(UInt<8>("ha7"), i), golden_167, _GEN_166) @[main.scala 77:32 main.scala 77:32]
    node _GEN_168 = mux(eq(UInt<8>("ha8"), i), golden_168, _GEN_167) @[main.scala 77:32 main.scala 77:32]
    node _GEN_169 = mux(eq(UInt<8>("ha9"), i), golden_169, _GEN_168) @[main.scala 77:32 main.scala 77:32]
    node _GEN_170 = mux(eq(UInt<8>("haa"), i), golden_170, _GEN_169) @[main.scala 77:32 main.scala 77:32]
    node _GEN_171 = mux(eq(UInt<8>("hab"), i), golden_171, _GEN_170) @[main.scala 77:32 main.scala 77:32]
    node _GEN_172 = mux(eq(UInt<8>("hac"), i), golden_172, _GEN_171) @[main.scala 77:32 main.scala 77:32]
    node _GEN_173 = mux(eq(UInt<8>("had"), i), golden_173, _GEN_172) @[main.scala 77:32 main.scala 77:32]
    node _GEN_174 = mux(eq(UInt<8>("hae"), i), golden_174, _GEN_173) @[main.scala 77:32 main.scala 77:32]
    node _GEN_175 = mux(eq(UInt<8>("haf"), i), golden_175, _GEN_174) @[main.scala 77:32 main.scala 77:32]
    node _GEN_176 = mux(eq(UInt<8>("hb0"), i), golden_176, _GEN_175) @[main.scala 77:32 main.scala 77:32]
    node _GEN_177 = mux(eq(UInt<8>("hb1"), i), golden_177, _GEN_176) @[main.scala 77:32 main.scala 77:32]
    node _GEN_178 = mux(eq(UInt<8>("hb2"), i), golden_178, _GEN_177) @[main.scala 77:32 main.scala 77:32]
    node _GEN_179 = mux(eq(UInt<8>("hb3"), i), golden_179, _GEN_178) @[main.scala 77:32 main.scala 77:32]
    node _GEN_180 = mux(eq(UInt<8>("hb4"), i), golden_180, _GEN_179) @[main.scala 77:32 main.scala 77:32]
    node _GEN_181 = mux(eq(UInt<8>("hb5"), i), golden_181, _GEN_180) @[main.scala 77:32 main.scala 77:32]
    node _GEN_182 = mux(eq(UInt<8>("hb6"), i), golden_182, _GEN_181) @[main.scala 77:32 main.scala 77:32]
    node _GEN_183 = mux(eq(UInt<8>("hb7"), i), golden_183, _GEN_182) @[main.scala 77:32 main.scala 77:32]
    node _GEN_184 = mux(eq(UInt<8>("hb8"), i), golden_184, _GEN_183) @[main.scala 77:32 main.scala 77:32]
    node _GEN_185 = mux(eq(UInt<8>("hb9"), i), golden_185, _GEN_184) @[main.scala 77:32 main.scala 77:32]
    node _GEN_186 = mux(eq(UInt<8>("hba"), i), golden_186, _GEN_185) @[main.scala 77:32 main.scala 77:32]
    node _GEN_187 = mux(eq(UInt<8>("hbb"), i), golden_187, _GEN_186) @[main.scala 77:32 main.scala 77:32]
    node _GEN_188 = mux(eq(UInt<8>("hbc"), i), golden_188, _GEN_187) @[main.scala 77:32 main.scala 77:32]
    node _GEN_189 = mux(eq(UInt<8>("hbd"), i), golden_189, _GEN_188) @[main.scala 77:32 main.scala 77:32]
    node _GEN_190 = mux(eq(UInt<8>("hbe"), i), golden_190, _GEN_189) @[main.scala 77:32 main.scala 77:32]
    node _GEN_191 = mux(eq(UInt<8>("hbf"), i), golden_191, _GEN_190) @[main.scala 77:32 main.scala 77:32]
    node _GEN_192 = mux(eq(UInt<8>("hc0"), i), golden_192, _GEN_191) @[main.scala 77:32 main.scala 77:32]
    node _GEN_193 = mux(eq(UInt<8>("hc1"), i), golden_193, _GEN_192) @[main.scala 77:32 main.scala 77:32]
    node _GEN_194 = mux(eq(UInt<8>("hc2"), i), golden_194, _GEN_193) @[main.scala 77:32 main.scala 77:32]
    node _GEN_195 = mux(eq(UInt<8>("hc3"), i), golden_195, _GEN_194) @[main.scala 77:32 main.scala 77:32]
    node _GEN_196 = mux(eq(UInt<8>("hc4"), i), golden_196, _GEN_195) @[main.scala 77:32 main.scala 77:32]
    node _GEN_197 = mux(eq(UInt<8>("hc5"), i), golden_197, _GEN_196) @[main.scala 77:32 main.scala 77:32]
    node _GEN_198 = mux(eq(UInt<8>("hc6"), i), golden_198, _GEN_197) @[main.scala 77:32 main.scala 77:32]
    node _GEN_199 = mux(eq(UInt<8>("hc7"), i), golden_199, _GEN_198) @[main.scala 77:32 main.scala 77:32]
    node _GEN_200 = mux(eq(UInt<8>("hc8"), i), golden_200, _GEN_199) @[main.scala 77:32 main.scala 77:32]
    node _GEN_201 = mux(eq(UInt<8>("hc9"), i), golden_201, _GEN_200) @[main.scala 77:32 main.scala 77:32]
    node _GEN_202 = mux(eq(UInt<8>("hca"), i), golden_202, _GEN_201) @[main.scala 77:32 main.scala 77:32]
    node _GEN_203 = mux(eq(UInt<8>("hcb"), i), golden_203, _GEN_202) @[main.scala 77:32 main.scala 77:32]
    node _GEN_204 = mux(eq(UInt<8>("hcc"), i), golden_204, _GEN_203) @[main.scala 77:32 main.scala 77:32]
    node _GEN_205 = mux(eq(UInt<8>("hcd"), i), golden_205, _GEN_204) @[main.scala 77:32 main.scala 77:32]
    node _GEN_206 = mux(eq(UInt<8>("hce"), i), golden_206, _GEN_205) @[main.scala 77:32 main.scala 77:32]
    node _GEN_207 = mux(eq(UInt<8>("hcf"), i), golden_207, _GEN_206) @[main.scala 77:32 main.scala 77:32]
    node _GEN_208 = mux(eq(UInt<8>("hd0"), i), golden_208, _GEN_207) @[main.scala 77:32 main.scala 77:32]
    node _GEN_209 = mux(eq(UInt<8>("hd1"), i), golden_209, _GEN_208) @[main.scala 77:32 main.scala 77:32]
    node _GEN_210 = mux(eq(UInt<8>("hd2"), i), golden_210, _GEN_209) @[main.scala 77:32 main.scala 77:32]
    node _GEN_211 = mux(eq(UInt<8>("hd3"), i), golden_211, _GEN_210) @[main.scala 77:32 main.scala 77:32]
    node _GEN_212 = mux(eq(UInt<8>("hd4"), i), golden_212, _GEN_211) @[main.scala 77:32 main.scala 77:32]
    node _GEN_213 = mux(eq(UInt<8>("hd5"), i), golden_213, _GEN_212) @[main.scala 77:32 main.scala 77:32]
    node _GEN_214 = mux(eq(UInt<8>("hd6"), i), golden_214, _GEN_213) @[main.scala 77:32 main.scala 77:32]
    node _GEN_215 = mux(eq(UInt<8>("hd7"), i), golden_215, _GEN_214) @[main.scala 77:32 main.scala 77:32]
    node _GEN_216 = mux(eq(UInt<8>("hd8"), i), golden_216, _GEN_215) @[main.scala 77:32 main.scala 77:32]
    node _GEN_217 = mux(eq(UInt<8>("hd9"), i), golden_217, _GEN_216) @[main.scala 77:32 main.scala 77:32]
    node _GEN_218 = mux(eq(UInt<8>("hda"), i), golden_218, _GEN_217) @[main.scala 77:32 main.scala 77:32]
    node _GEN_219 = mux(eq(UInt<8>("hdb"), i), golden_219, _GEN_218) @[main.scala 77:32 main.scala 77:32]
    node _GEN_220 = mux(eq(UInt<8>("hdc"), i), golden_220, _GEN_219) @[main.scala 77:32 main.scala 77:32]
    node _GEN_221 = mux(eq(UInt<8>("hdd"), i), golden_221, _GEN_220) @[main.scala 77:32 main.scala 77:32]
    node _GEN_222 = mux(eq(UInt<8>("hde"), i), golden_222, _GEN_221) @[main.scala 77:32 main.scala 77:32]
    node _GEN_223 = mux(eq(UInt<8>("hdf"), i), golden_223, _GEN_222) @[main.scala 77:32 main.scala 77:32]
    node _GEN_224 = mux(eq(UInt<8>("he0"), i), golden_224, _GEN_223) @[main.scala 77:32 main.scala 77:32]
    node _GEN_225 = mux(eq(UInt<8>("he1"), i), golden_225, _GEN_224) @[main.scala 77:32 main.scala 77:32]
    node _GEN_226 = mux(eq(UInt<8>("he2"), i), golden_226, _GEN_225) @[main.scala 77:32 main.scala 77:32]
    node _GEN_227 = mux(eq(UInt<8>("he3"), i), golden_227, _GEN_226) @[main.scala 77:32 main.scala 77:32]
    node _GEN_228 = mux(eq(UInt<8>("he4"), i), golden_228, _GEN_227) @[main.scala 77:32 main.scala 77:32]
    node _GEN_229 = mux(eq(UInt<8>("he5"), i), golden_229, _GEN_228) @[main.scala 77:32 main.scala 77:32]
    node _GEN_230 = mux(eq(UInt<8>("he6"), i), golden_230, _GEN_229) @[main.scala 77:32 main.scala 77:32]
    node _GEN_231 = mux(eq(UInt<8>("he7"), i), golden_231, _GEN_230) @[main.scala 77:32 main.scala 77:32]
    node _GEN_232 = mux(eq(UInt<8>("he8"), i), golden_232, _GEN_231) @[main.scala 77:32 main.scala 77:32]
    node _GEN_233 = mux(eq(UInt<8>("he9"), i), golden_233, _GEN_232) @[main.scala 77:32 main.scala 77:32]
    node _GEN_234 = mux(eq(UInt<8>("hea"), i), golden_234, _GEN_233) @[main.scala 77:32 main.scala 77:32]
    node _GEN_235 = mux(eq(UInt<8>("heb"), i), golden_235, _GEN_234) @[main.scala 77:32 main.scala 77:32]
    node _GEN_236 = mux(eq(UInt<8>("hec"), i), golden_236, _GEN_235) @[main.scala 77:32 main.scala 77:32]
    node _GEN_237 = mux(eq(UInt<8>("hed"), i), golden_237, _GEN_236) @[main.scala 77:32 main.scala 77:32]
    node _GEN_238 = mux(eq(UInt<8>("hee"), i), golden_238, _GEN_237) @[main.scala 77:32 main.scala 77:32]
    node _GEN_239 = mux(eq(UInt<8>("hef"), i), golden_239, _GEN_238) @[main.scala 77:32 main.scala 77:32]
    node _GEN_240 = mux(eq(UInt<8>("hf0"), i), golden_240, _GEN_239) @[main.scala 77:32 main.scala 77:32]
    node _GEN_241 = mux(eq(UInt<8>("hf1"), i), golden_241, _GEN_240) @[main.scala 77:32 main.scala 77:32]
    node _GEN_242 = mux(eq(UInt<8>("hf2"), i), golden_242, _GEN_241) @[main.scala 77:32 main.scala 77:32]
    node _GEN_243 = mux(eq(UInt<8>("hf3"), i), golden_243, _GEN_242) @[main.scala 77:32 main.scala 77:32]
    node _GEN_244 = mux(eq(UInt<8>("hf4"), i), golden_244, _GEN_243) @[main.scala 77:32 main.scala 77:32]
    node _GEN_245 = mux(eq(UInt<8>("hf5"), i), golden_245, _GEN_244) @[main.scala 77:32 main.scala 77:32]
    node _GEN_246 = mux(eq(UInt<8>("hf6"), i), golden_246, _GEN_245) @[main.scala 77:32 main.scala 77:32]
    node _GEN_247 = mux(eq(UInt<8>("hf7"), i), golden_247, _GEN_246) @[main.scala 77:32 main.scala 77:32]
    node _GEN_248 = mux(eq(UInt<8>("hf8"), i), golden_248, _GEN_247) @[main.scala 77:32 main.scala 77:32]
    node _GEN_249 = mux(eq(UInt<8>("hf9"), i), golden_249, _GEN_248) @[main.scala 77:32 main.scala 77:32]
    node _GEN_250 = mux(eq(UInt<8>("hfa"), i), golden_250, _GEN_249) @[main.scala 77:32 main.scala 77:32]
    node _GEN_251 = mux(eq(UInt<8>("hfb"), i), golden_251, _GEN_250) @[main.scala 77:32 main.scala 77:32]
    node _GEN_252 = mux(eq(UInt<8>("hfc"), i), golden_252, _GEN_251) @[main.scala 77:32 main.scala 77:32]
    node _GEN_253 = mux(eq(UInt<8>("hfd"), i), golden_253, _GEN_252) @[main.scala 77:32 main.scala 77:32]
    node _GEN_254 = mux(eq(UInt<8>("hfe"), i), golden_254, _GEN_253) @[main.scala 77:32 main.scala 77:32]
    node _GEN_255 = mux(eq(UInt<8>("hff"), i), golden_255, _GEN_254) @[main.scala 77:32 main.scala 77:32]
    node _golden_i = _GEN_255 @[main.scala 77:32]
    node _T_11 = eq(iLFSR.io_out, _golden_i) @[main.scala 77:32]
    node _T_12 = and(pass, _T_11) @[main.scala 77:16]
    node _T_13 = bits(reset, 0, 0) @[main.scala 78:9]
    node _T_14 = eq(_T_13, UInt<1>("h0")) @[main.scala 78:9]
    node _WIRE_0 = pad(UInt<1>("h1"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_1 = pad(UInt<2>("h2"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_2 = pad(UInt<3>("h4"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_3 = pad(UInt<4>("h8"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_4 = pad(UInt<5>("h10"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_5 = UInt<6>("h21") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_6 = pad(UInt<2>("h3"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_7 = pad(UInt<3>("h6"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_8 = pad(UInt<4>("hc"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_9 = pad(UInt<5>("h18"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_10 = UInt<6>("h31") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_11 = UInt<6>("h22") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_12 = pad(UInt<3>("h5"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_13 = pad(UInt<4>("ha"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_14 = pad(UInt<5>("h14"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_15 = UInt<6>("h29") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_16 = pad(UInt<5>("h13"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_17 = UInt<6>("h27") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_18 = pad(UInt<4>("hf"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_19 = pad(UInt<5>("h1e"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_20 = UInt<6>("h3d") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_21 = UInt<6>("h3a") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_22 = UInt<6>("h34") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_23 = UInt<6>("h28") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_24 = pad(UInt<5>("h11"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_25 = UInt<6>("h23") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_26 = pad(UInt<3>("h7"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_27 = pad(UInt<4>("he"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_28 = pad(UInt<5>("h1c"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_29 = UInt<6>("h39") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_30 = UInt<6>("h32") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_31 = UInt<6>("h24") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_32 = pad(UInt<4>("h9"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_33 = pad(UInt<5>("h12"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_34 = UInt<6>("h25") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_35 = pad(UInt<4>("hb"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_36 = pad(UInt<5>("h16"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_37 = UInt<6>("h2d") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_38 = pad(UInt<5>("h1b"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_39 = UInt<6>("h37") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_40 = UInt<6>("h2e") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_41 = pad(UInt<5>("h1d"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_42 = UInt<6>("h3b") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_43 = UInt<6>("h36") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_44 = UInt<6>("h2c") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_45 = pad(UInt<5>("h19"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_46 = UInt<6>("h33") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_47 = UInt<6>("h26") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_48 = pad(UInt<4>("hd"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_49 = pad(UInt<5>("h1a"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_50 = UInt<6>("h35") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_51 = UInt<6>("h2a") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_52 = pad(UInt<5>("h15"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_53 = UInt<6>("h2b") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_54 = pad(UInt<5>("h17"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_55 = UInt<6>("h2f") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_56 = pad(UInt<5>("h1f"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_57 = UInt<6>("h3f") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_58 = UInt<6>("h3e") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_59 = UInt<6>("h3c") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_60 = UInt<6>("h38") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_61 = UInt<6>("h30") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_62 = UInt<6>("h20") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_63 = pad(UInt<1>("h1"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_64 = pad(UInt<2>("h2"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_65 = pad(UInt<3>("h4"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_66 = pad(UInt<4>("h8"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_67 = pad(UInt<5>("h10"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_68 = UInt<6>("h21") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_69 = pad(UInt<2>("h3"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_70 = pad(UInt<3>("h6"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_71 = pad(UInt<4>("hc"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_72 = pad(UInt<5>("h18"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_73 = UInt<6>("h31") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_74 = UInt<6>("h22") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_75 = pad(UInt<3>("h5"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_76 = pad(UInt<4>("ha"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_77 = pad(UInt<5>("h14"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_78 = UInt<6>("h29") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_79 = pad(UInt<5>("h13"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_80 = UInt<6>("h27") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_81 = pad(UInt<4>("hf"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_82 = pad(UInt<5>("h1e"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_83 = UInt<6>("h3d") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_84 = UInt<6>("h3a") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_85 = UInt<6>("h34") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_86 = UInt<6>("h28") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_87 = pad(UInt<5>("h11"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_88 = UInt<6>("h23") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_89 = pad(UInt<3>("h7"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_90 = pad(UInt<4>("he"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_91 = pad(UInt<5>("h1c"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_92 = UInt<6>("h39") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_93 = UInt<6>("h32") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_94 = UInt<6>("h24") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_95 = pad(UInt<4>("h9"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_96 = pad(UInt<5>("h12"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_97 = UInt<6>("h25") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_98 = pad(UInt<4>("hb"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_99 = pad(UInt<5>("h16"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_100 = UInt<6>("h2d") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_101 = pad(UInt<5>("h1b"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_102 = UInt<6>("h37") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_103 = UInt<6>("h2e") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_104 = pad(UInt<5>("h1d"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_105 = UInt<6>("h3b") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_106 = UInt<6>("h36") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_107 = UInt<6>("h2c") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_108 = pad(UInt<5>("h19"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_109 = UInt<6>("h33") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_110 = UInt<6>("h26") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_111 = pad(UInt<4>("hd"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_112 = pad(UInt<5>("h1a"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_113 = UInt<6>("h35") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_114 = UInt<6>("h2a") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_115 = pad(UInt<5>("h15"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_116 = UInt<6>("h2b") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_117 = pad(UInt<5>("h17"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_118 = UInt<6>("h2f") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_119 = pad(UInt<5>("h1f"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_120 = UInt<6>("h3f") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_121 = UInt<6>("h3e") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_122 = UInt<6>("h3c") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_123 = UInt<6>("h38") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_124 = UInt<6>("h30") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_125 = UInt<6>("h20") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_126 = pad(UInt<1>("h1"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_127 = pad(UInt<2>("h2"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_128 = pad(UInt<3>("h4"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_129 = pad(UInt<4>("h8"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_130 = pad(UInt<5>("h10"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_131 = UInt<6>("h21") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_132 = pad(UInt<2>("h3"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_133 = pad(UInt<3>("h6"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_134 = pad(UInt<4>("hc"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_135 = pad(UInt<5>("h18"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_136 = UInt<6>("h31") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_137 = UInt<6>("h22") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_138 = pad(UInt<3>("h5"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_139 = pad(UInt<4>("ha"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_140 = pad(UInt<5>("h14"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_141 = UInt<6>("h29") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_142 = pad(UInt<5>("h13"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_143 = UInt<6>("h27") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_144 = pad(UInt<4>("hf"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_145 = pad(UInt<5>("h1e"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_146 = UInt<6>("h3d") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_147 = UInt<6>("h3a") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_148 = UInt<6>("h34") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_149 = UInt<6>("h28") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_150 = pad(UInt<5>("h11"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_151 = UInt<6>("h23") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_152 = pad(UInt<3>("h7"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_153 = pad(UInt<4>("he"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_154 = pad(UInt<5>("h1c"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_155 = UInt<6>("h39") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_156 = UInt<6>("h32") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_157 = UInt<6>("h24") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_158 = pad(UInt<4>("h9"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_159 = pad(UInt<5>("h12"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_160 = UInt<6>("h25") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_161 = pad(UInt<4>("hb"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_162 = pad(UInt<5>("h16"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_163 = UInt<6>("h2d") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_164 = pad(UInt<5>("h1b"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_165 = UInt<6>("h37") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_166 = UInt<6>("h2e") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_167 = pad(UInt<5>("h1d"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_168 = UInt<6>("h3b") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_169 = UInt<6>("h36") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_170 = UInt<6>("h2c") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_171 = pad(UInt<5>("h19"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_172 = UInt<6>("h33") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_173 = UInt<6>("h26") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_174 = pad(UInt<4>("hd"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_175 = pad(UInt<5>("h1a"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_176 = UInt<6>("h35") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_177 = UInt<6>("h2a") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_178 = pad(UInt<5>("h15"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_179 = UInt<6>("h2b") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_180 = pad(UInt<5>("h17"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_181 = UInt<6>("h2f") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_182 = pad(UInt<5>("h1f"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_183 = UInt<6>("h3f") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_184 = UInt<6>("h3e") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_185 = UInt<6>("h3c") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_186 = UInt<6>("h38") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_187 = UInt<6>("h30") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_188 = UInt<6>("h20") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_189 = pad(UInt<1>("h1"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_190 = pad(UInt<2>("h2"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_191 = pad(UInt<3>("h4"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_192 = pad(UInt<4>("h8"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_193 = pad(UInt<5>("h10"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_194 = UInt<6>("h21") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_195 = pad(UInt<2>("h3"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_196 = pad(UInt<3>("h6"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_197 = pad(UInt<4>("hc"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_198 = pad(UInt<5>("h18"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_199 = UInt<6>("h31") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_200 = UInt<6>("h22") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_201 = pad(UInt<3>("h5"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_202 = pad(UInt<4>("ha"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_203 = pad(UInt<5>("h14"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_204 = UInt<6>("h29") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_205 = pad(UInt<5>("h13"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_206 = UInt<6>("h27") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_207 = pad(UInt<4>("hf"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_208 = pad(UInt<5>("h1e"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_209 = UInt<6>("h3d") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_210 = UInt<6>("h3a") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_211 = UInt<6>("h34") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_212 = UInt<6>("h28") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_213 = pad(UInt<5>("h11"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_214 = UInt<6>("h23") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_215 = pad(UInt<3>("h7"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_216 = pad(UInt<4>("he"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_217 = pad(UInt<5>("h1c"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_218 = UInt<6>("h39") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_219 = UInt<6>("h32") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_220 = UInt<6>("h24") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_221 = pad(UInt<4>("h9"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_222 = pad(UInt<5>("h12"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_223 = UInt<6>("h25") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_224 = pad(UInt<4>("hb"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_225 = pad(UInt<5>("h16"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_226 = UInt<6>("h2d") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_227 = pad(UInt<5>("h1b"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_228 = UInt<6>("h37") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_229 = UInt<6>("h2e") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_230 = pad(UInt<5>("h1d"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_231 = UInt<6>("h3b") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_232 = UInt<6>("h36") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_233 = UInt<6>("h2c") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_234 = pad(UInt<5>("h19"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_235 = UInt<6>("h33") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_236 = UInt<6>("h26") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_237 = pad(UInt<4>("hd"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_238 = pad(UInt<5>("h1a"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_239 = UInt<6>("h35") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_240 = UInt<6>("h2a") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_241 = pad(UInt<5>("h15"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_242 = UInt<6>("h2b") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_243 = pad(UInt<5>("h17"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_244 = UInt<6>("h2f") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_245 = pad(UInt<5>("h1f"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_246 = UInt<6>("h3f") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_247 = UInt<6>("h3e") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_248 = UInt<6>("h3c") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_249 = UInt<6>("h38") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_250 = UInt<6>("h30") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_251 = UInt<6>("h20") @[main.scala 33:31 main.scala 33:31]
    node _WIRE_252 = pad(UInt<1>("h1"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_253 = pad(UInt<2>("h2"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_254 = pad(UInt<3>("h4"), 6) @[main.scala 33:31 main.scala 33:31]
    node _WIRE_255 = pad(UInt<4>("h8"), 6) @[main.scala 33:31 main.scala 33:31]
    iLFSR.clock <= clock
    iLFSR.reset <= reset
    i <= mux(reset, UInt<8>("h0"), _T_10) @[main.scala 32:18 main.scala 32:18 main.scala 76:5]
    golden_0 <= mux(reset, _WIRE_0, golden_0) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_1 <= mux(reset, _WIRE_1, golden_1) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_2 <= mux(reset, _WIRE_2, golden_2) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_3 <= mux(reset, _WIRE_3, golden_3) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_4 <= mux(reset, _WIRE_4, golden_4) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_5 <= mux(reset, _WIRE_5, golden_5) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_6 <= mux(reset, _WIRE_6, golden_6) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_7 <= mux(reset, _WIRE_7, golden_7) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_8 <= mux(reset, _WIRE_8, golden_8) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_9 <= mux(reset, _WIRE_9, golden_9) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_10 <= mux(reset, _WIRE_10, golden_10) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_11 <= mux(reset, _WIRE_11, golden_11) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_12 <= mux(reset, _WIRE_12, golden_12) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_13 <= mux(reset, _WIRE_13, golden_13) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_14 <= mux(reset, _WIRE_14, golden_14) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_15 <= mux(reset, _WIRE_15, golden_15) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_16 <= mux(reset, _WIRE_16, golden_16) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_17 <= mux(reset, _WIRE_17, golden_17) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_18 <= mux(reset, _WIRE_18, golden_18) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_19 <= mux(reset, _WIRE_19, golden_19) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_20 <= mux(reset, _WIRE_20, golden_20) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_21 <= mux(reset, _WIRE_21, golden_21) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_22 <= mux(reset, _WIRE_22, golden_22) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_23 <= mux(reset, _WIRE_23, golden_23) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_24 <= mux(reset, _WIRE_24, golden_24) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_25 <= mux(reset, _WIRE_25, golden_25) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_26 <= mux(reset, _WIRE_26, golden_26) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_27 <= mux(reset, _WIRE_27, golden_27) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_28 <= mux(reset, _WIRE_28, golden_28) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_29 <= mux(reset, _WIRE_29, golden_29) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_30 <= mux(reset, _WIRE_30, golden_30) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_31 <= mux(reset, _WIRE_31, golden_31) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_32 <= mux(reset, _WIRE_32, golden_32) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_33 <= mux(reset, _WIRE_33, golden_33) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_34 <= mux(reset, _WIRE_34, golden_34) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_35 <= mux(reset, _WIRE_35, golden_35) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_36 <= mux(reset, _WIRE_36, golden_36) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_37 <= mux(reset, _WIRE_37, golden_37) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_38 <= mux(reset, _WIRE_38, golden_38) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_39 <= mux(reset, _WIRE_39, golden_39) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_40 <= mux(reset, _WIRE_40, golden_40) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_41 <= mux(reset, _WIRE_41, golden_41) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_42 <= mux(reset, _WIRE_42, golden_42) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_43 <= mux(reset, _WIRE_43, golden_43) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_44 <= mux(reset, _WIRE_44, golden_44) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_45 <= mux(reset, _WIRE_45, golden_45) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_46 <= mux(reset, _WIRE_46, golden_46) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_47 <= mux(reset, _WIRE_47, golden_47) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_48 <= mux(reset, _WIRE_48, golden_48) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_49 <= mux(reset, _WIRE_49, golden_49) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_50 <= mux(reset, _WIRE_50, golden_50) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_51 <= mux(reset, _WIRE_51, golden_51) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_52 <= mux(reset, _WIRE_52, golden_52) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_53 <= mux(reset, _WIRE_53, golden_53) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_54 <= mux(reset, _WIRE_54, golden_54) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_55 <= mux(reset, _WIRE_55, golden_55) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_56 <= mux(reset, _WIRE_56, golden_56) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_57 <= mux(reset, _WIRE_57, golden_57) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_58 <= mux(reset, _WIRE_58, golden_58) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_59 <= mux(reset, _WIRE_59, golden_59) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_60 <= mux(reset, _WIRE_60, golden_60) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_61 <= mux(reset, _WIRE_61, golden_61) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_62 <= mux(reset, _WIRE_62, golden_62) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_63 <= mux(reset, _WIRE_63, golden_63) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_64 <= mux(reset, _WIRE_64, golden_64) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_65 <= mux(reset, _WIRE_65, golden_65) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_66 <= mux(reset, _WIRE_66, golden_66) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_67 <= mux(reset, _WIRE_67, golden_67) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_68 <= mux(reset, _WIRE_68, golden_68) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_69 <= mux(reset, _WIRE_69, golden_69) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_70 <= mux(reset, _WIRE_70, golden_70) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_71 <= mux(reset, _WIRE_71, golden_71) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_72 <= mux(reset, _WIRE_72, golden_72) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_73 <= mux(reset, _WIRE_73, golden_73) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_74 <= mux(reset, _WIRE_74, golden_74) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_75 <= mux(reset, _WIRE_75, golden_75) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_76 <= mux(reset, _WIRE_76, golden_76) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_77 <= mux(reset, _WIRE_77, golden_77) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_78 <= mux(reset, _WIRE_78, golden_78) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_79 <= mux(reset, _WIRE_79, golden_79) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_80 <= mux(reset, _WIRE_80, golden_80) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_81 <= mux(reset, _WIRE_81, golden_81) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_82 <= mux(reset, _WIRE_82, golden_82) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_83 <= mux(reset, _WIRE_83, golden_83) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_84 <= mux(reset, _WIRE_84, golden_84) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_85 <= mux(reset, _WIRE_85, golden_85) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_86 <= mux(reset, _WIRE_86, golden_86) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_87 <= mux(reset, _WIRE_87, golden_87) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_88 <= mux(reset, _WIRE_88, golden_88) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_89 <= mux(reset, _WIRE_89, golden_89) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_90 <= mux(reset, _WIRE_90, golden_90) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_91 <= mux(reset, _WIRE_91, golden_91) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_92 <= mux(reset, _WIRE_92, golden_92) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_93 <= mux(reset, _WIRE_93, golden_93) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_94 <= mux(reset, _WIRE_94, golden_94) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_95 <= mux(reset, _WIRE_95, golden_95) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_96 <= mux(reset, _WIRE_96, golden_96) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_97 <= mux(reset, _WIRE_97, golden_97) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_98 <= mux(reset, _WIRE_98, golden_98) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_99 <= mux(reset, _WIRE_99, golden_99) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_100 <= mux(reset, _WIRE_100, golden_100) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_101 <= mux(reset, _WIRE_101, golden_101) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_102 <= mux(reset, _WIRE_102, golden_102) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_103 <= mux(reset, _WIRE_103, golden_103) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_104 <= mux(reset, _WIRE_104, golden_104) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_105 <= mux(reset, _WIRE_105, golden_105) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_106 <= mux(reset, _WIRE_106, golden_106) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_107 <= mux(reset, _WIRE_107, golden_107) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_108 <= mux(reset, _WIRE_108, golden_108) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_109 <= mux(reset, _WIRE_109, golden_109) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_110 <= mux(reset, _WIRE_110, golden_110) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_111 <= mux(reset, _WIRE_111, golden_111) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_112 <= mux(reset, _WIRE_112, golden_112) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_113 <= mux(reset, _WIRE_113, golden_113) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_114 <= mux(reset, _WIRE_114, golden_114) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_115 <= mux(reset, _WIRE_115, golden_115) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_116 <= mux(reset, _WIRE_116, golden_116) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_117 <= mux(reset, _WIRE_117, golden_117) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_118 <= mux(reset, _WIRE_118, golden_118) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_119 <= mux(reset, _WIRE_119, golden_119) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_120 <= mux(reset, _WIRE_120, golden_120) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_121 <= mux(reset, _WIRE_121, golden_121) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_122 <= mux(reset, _WIRE_122, golden_122) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_123 <= mux(reset, _WIRE_123, golden_123) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_124 <= mux(reset, _WIRE_124, golden_124) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_125 <= mux(reset, _WIRE_125, golden_125) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_126 <= mux(reset, _WIRE_126, golden_126) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_127 <= mux(reset, _WIRE_127, golden_127) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_128 <= mux(reset, _WIRE_128, golden_128) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_129 <= mux(reset, _WIRE_129, golden_129) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_130 <= mux(reset, _WIRE_130, golden_130) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_131 <= mux(reset, _WIRE_131, golden_131) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_132 <= mux(reset, _WIRE_132, golden_132) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_133 <= mux(reset, _WIRE_133, golden_133) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_134 <= mux(reset, _WIRE_134, golden_134) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_135 <= mux(reset, _WIRE_135, golden_135) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_136 <= mux(reset, _WIRE_136, golden_136) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_137 <= mux(reset, _WIRE_137, golden_137) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_138 <= mux(reset, _WIRE_138, golden_138) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_139 <= mux(reset, _WIRE_139, golden_139) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_140 <= mux(reset, _WIRE_140, golden_140) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_141 <= mux(reset, _WIRE_141, golden_141) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_142 <= mux(reset, _WIRE_142, golden_142) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_143 <= mux(reset, _WIRE_143, golden_143) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_144 <= mux(reset, _WIRE_144, golden_144) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_145 <= mux(reset, _WIRE_145, golden_145) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_146 <= mux(reset, _WIRE_146, golden_146) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_147 <= mux(reset, _WIRE_147, golden_147) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_148 <= mux(reset, _WIRE_148, golden_148) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_149 <= mux(reset, _WIRE_149, golden_149) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_150 <= mux(reset, _WIRE_150, golden_150) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_151 <= mux(reset, _WIRE_151, golden_151) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_152 <= mux(reset, _WIRE_152, golden_152) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_153 <= mux(reset, _WIRE_153, golden_153) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_154 <= mux(reset, _WIRE_154, golden_154) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_155 <= mux(reset, _WIRE_155, golden_155) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_156 <= mux(reset, _WIRE_156, golden_156) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_157 <= mux(reset, _WIRE_157, golden_157) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_158 <= mux(reset, _WIRE_158, golden_158) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_159 <= mux(reset, _WIRE_159, golden_159) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_160 <= mux(reset, _WIRE_160, golden_160) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_161 <= mux(reset, _WIRE_161, golden_161) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_162 <= mux(reset, _WIRE_162, golden_162) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_163 <= mux(reset, _WIRE_163, golden_163) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_164 <= mux(reset, _WIRE_164, golden_164) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_165 <= mux(reset, _WIRE_165, golden_165) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_166 <= mux(reset, _WIRE_166, golden_166) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_167 <= mux(reset, _WIRE_167, golden_167) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_168 <= mux(reset, _WIRE_168, golden_168) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_169 <= mux(reset, _WIRE_169, golden_169) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_170 <= mux(reset, _WIRE_170, golden_170) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_171 <= mux(reset, _WIRE_171, golden_171) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_172 <= mux(reset, _WIRE_172, golden_172) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_173 <= mux(reset, _WIRE_173, golden_173) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_174 <= mux(reset, _WIRE_174, golden_174) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_175 <= mux(reset, _WIRE_175, golden_175) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_176 <= mux(reset, _WIRE_176, golden_176) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_177 <= mux(reset, _WIRE_177, golden_177) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_178 <= mux(reset, _WIRE_178, golden_178) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_179 <= mux(reset, _WIRE_179, golden_179) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_180 <= mux(reset, _WIRE_180, golden_180) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_181 <= mux(reset, _WIRE_181, golden_181) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_182 <= mux(reset, _WIRE_182, golden_182) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_183 <= mux(reset, _WIRE_183, golden_183) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_184 <= mux(reset, _WIRE_184, golden_184) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_185 <= mux(reset, _WIRE_185, golden_185) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_186 <= mux(reset, _WIRE_186, golden_186) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_187 <= mux(reset, _WIRE_187, golden_187) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_188 <= mux(reset, _WIRE_188, golden_188) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_189 <= mux(reset, _WIRE_189, golden_189) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_190 <= mux(reset, _WIRE_190, golden_190) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_191 <= mux(reset, _WIRE_191, golden_191) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_192 <= mux(reset, _WIRE_192, golden_192) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_193 <= mux(reset, _WIRE_193, golden_193) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_194 <= mux(reset, _WIRE_194, golden_194) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_195 <= mux(reset, _WIRE_195, golden_195) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_196 <= mux(reset, _WIRE_196, golden_196) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_197 <= mux(reset, _WIRE_197, golden_197) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_198 <= mux(reset, _WIRE_198, golden_198) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_199 <= mux(reset, _WIRE_199, golden_199) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_200 <= mux(reset, _WIRE_200, golden_200) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_201 <= mux(reset, _WIRE_201, golden_201) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_202 <= mux(reset, _WIRE_202, golden_202) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_203 <= mux(reset, _WIRE_203, golden_203) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_204 <= mux(reset, _WIRE_204, golden_204) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_205 <= mux(reset, _WIRE_205, golden_205) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_206 <= mux(reset, _WIRE_206, golden_206) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_207 <= mux(reset, _WIRE_207, golden_207) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_208 <= mux(reset, _WIRE_208, golden_208) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_209 <= mux(reset, _WIRE_209, golden_209) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_210 <= mux(reset, _WIRE_210, golden_210) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_211 <= mux(reset, _WIRE_211, golden_211) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_212 <= mux(reset, _WIRE_212, golden_212) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_213 <= mux(reset, _WIRE_213, golden_213) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_214 <= mux(reset, _WIRE_214, golden_214) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_215 <= mux(reset, _WIRE_215, golden_215) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_216 <= mux(reset, _WIRE_216, golden_216) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_217 <= mux(reset, _WIRE_217, golden_217) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_218 <= mux(reset, _WIRE_218, golden_218) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_219 <= mux(reset, _WIRE_219, golden_219) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_220 <= mux(reset, _WIRE_220, golden_220) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_221 <= mux(reset, _WIRE_221, golden_221) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_222 <= mux(reset, _WIRE_222, golden_222) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_223 <= mux(reset, _WIRE_223, golden_223) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_224 <= mux(reset, _WIRE_224, golden_224) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_225 <= mux(reset, _WIRE_225, golden_225) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_226 <= mux(reset, _WIRE_226, golden_226) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_227 <= mux(reset, _WIRE_227, golden_227) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_228 <= mux(reset, _WIRE_228, golden_228) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_229 <= mux(reset, _WIRE_229, golden_229) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_230 <= mux(reset, _WIRE_230, golden_230) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_231 <= mux(reset, _WIRE_231, golden_231) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_232 <= mux(reset, _WIRE_232, golden_232) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_233 <= mux(reset, _WIRE_233, golden_233) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_234 <= mux(reset, _WIRE_234, golden_234) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_235 <= mux(reset, _WIRE_235, golden_235) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_236 <= mux(reset, _WIRE_236, golden_236) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_237 <= mux(reset, _WIRE_237, golden_237) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_238 <= mux(reset, _WIRE_238, golden_238) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_239 <= mux(reset, _WIRE_239, golden_239) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_240 <= mux(reset, _WIRE_240, golden_240) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_241 <= mux(reset, _WIRE_241, golden_241) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_242 <= mux(reset, _WIRE_242, golden_242) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_243 <= mux(reset, _WIRE_243, golden_243) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_244 <= mux(reset, _WIRE_244, golden_244) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_245 <= mux(reset, _WIRE_245, golden_245) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_246 <= mux(reset, _WIRE_246, golden_246) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_247 <= mux(reset, _WIRE_247, golden_247) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_248 <= mux(reset, _WIRE_248, golden_248) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_249 <= mux(reset, _WIRE_249, golden_249) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_250 <= mux(reset, _WIRE_250, golden_250) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_251 <= mux(reset, _WIRE_251, golden_251) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_252 <= mux(reset, _WIRE_252, golden_252) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_253 <= mux(reset, _WIRE_253, golden_253) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_254 <= mux(reset, _WIRE_254, golden_254) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    golden_255 <= mux(reset, _WIRE_255, golden_255) @[main.scala 33:23 main.scala 33:23 main.scala 33:23]
    pass <= mux(reset, UInt<1>("h1"), _T_12) @[main.scala 66:21 main.scala 66:21 main.scala 77:8]
    printf(clock, and(and(and(and(UInt<1>("h1"), _T), _T_1), _T_3), UInt<1>("h1")), "PASS\n") @[main.scala 69:13]
    printf(clock, and(and(and(and(UInt<1>("h1"), _T), _T_4), _T_6), UInt<1>("h1")), "FAIL\n") @[main.scala 72:13]
    stop(clock, and(and(and(UInt<1>("h1"), _T), _T_8), UInt<1>("h1")), 0) @[main.scala 74:9]
