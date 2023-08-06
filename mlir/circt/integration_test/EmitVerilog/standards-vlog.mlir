// REQUIRES: questa

// RUN: firtool --verilog %s > %t1.1995.v
// RUN: firtool --verilog %s > %t1.2001.v
// RUN: firtool --verilog %s > %t1.2005.v
// RUN: firtool --verilog %s > %t1.2005.sv
// RUN: firtool --verilog %s > %t1.2009.sv
// RUN: firtool --verilog %s > %t1.2012.sv
// RUN: firtool --verilog %s> %t1.2017.sv

// RUN: vlog -lint %t1.1995.v -vlog95compat || true
// RUN: vlog -lint %t1.2001.v -vlog01compat || true
// RUN: vlog -lint %t1.2005.v || true
// RUN: vlog -lint -sv -sv05compat %t1.2005.sv
// RUN: vlog -lint -sv -sv09compat %t1.2009.sv
// RUN: vlog -lint -sv -sv12compat %t1.2012.sv
// RUN: vlog -lint -sv -sv17compat %t1.2017.sv 

hw.module @top(%clock : i1, %reset: i1,
                %a: i4, 
                %s: !hw.struct<foo: i2, bar: i4>,
                %parray: !hw.array<10xi4>,
                %uarray: !hw.uarray<16xi8>)
                 -> (r0: i4, r1: i4) {
  %0 = comb.or %a, %a : i4
  %1 = comb.and %a, %a : i4

  sv.always posedge %clock, negedge %reset {
  }

  sv.alwaysff(posedge %clock) {
    %fd = hw.constant 0x80000002 : i32
    sv.fwrite %fd, "Yo\n"
  }
  
  hw.output %0, %1 : i4, i4
}
