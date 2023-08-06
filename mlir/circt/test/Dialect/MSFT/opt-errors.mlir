// RUN: circt-opt %s --msft-discover-appids -verify-diagnostics -split-input-file

hw.module.extern @Foo()

hw.module @Top() {
  // expected-error @+1 {{Unknown device type 'WACKY'}}
  hw.instance "foo1" @Foo() -> () {"loc:" = #msft.physloc<WACKY, 0, 0, 0> } 
}

// -----

module {
  // expected-error @+1 {{unknown attribute `foo` in dialect `msft`}}
  hw.instance "foo1" @Foo() -> () {"loc:" = #msft.foo<""> } 
}

// -----

msft.module @M {} () {
  // expected-error @+1 {{Cannot find module definition 'Bar'}}
  msft.instance @instance @Bar () : () -> ()
  msft.output
}

// -----

msft.instance.hierarchy @reg {
  msft.instance.dynamic @reg::@reg {
    // expected-error @+1 {{'msft.pd.location' op cannot both have a global ref symbol and be a child of a dynamic instance op}}
    msft.pd.location @ref FF x: 0 y: 0 n: 0
  }
}

// -----

// expected-error @+1 {{'msft.pd.location' op must have either a global ref symbol of belong to a dynamic instance op}}
msft.pd.location FF x: 0 y: 0 n: 0

// -----

msft.module @M {} (%x : i32) {
  // expected-note @+1 {{first AppID located here}}
  comb.add %x, %x {msft.appid=#msft.appid<"add"[0]>} : i32
  // expected-error @+1 {{'comb.add' op Found multiple identical AppIDs in same module}}
  comb.add %x, %x {msft.appid=#msft.appid<"add"[0]>} : i32
  msft.output
}

// -----

msft.module @foo {} (%in0 : i32, %clk : i1) -> (out: i32) {
// expected-error @+1 {{'msft.hlc.linear' op expected only hw, comb, and msft dialect ops inside the datapath.}}
  %0 = msft.hlc.linear clock %clk : i32 {
    %c1_i1 = hw.constant 1 : i1
    %0 = seq.compreg %in0, %c1_i1 : i32
    msft.output %0 : i32
  }
  msft.output %0 : i32
}
