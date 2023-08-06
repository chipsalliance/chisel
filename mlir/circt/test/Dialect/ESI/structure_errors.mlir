// RUN: circt-opt %s -split-input-file --verify-diagnostics

msft.module @Foo {} (%in0: i1) -> (out: i1)
esi.pure_module @top {
  // expected-error @+1 {{'msft.instance' op instances in ESI pure modules can only contain channel ports}}
  %loopback = msft.instance @foo @Foo(%loopback) : (i1) -> (i1)
}

// -----

msft.module @Foo{} () -> (out: i1)

esi.pure_module @top {
  // expected-error @+1 {{'msft.instance' op instances in ESI pure modules can only contain channel ports}}
  msft.instance @foo @Foo() : () -> (i1)
}

// -----

msft.module @Foo {} (%in0 : !esi.channel<i1>) -> (out: !esi.channel<i1>)

esi.pure_module @top {
  %loopback = msft.instance @foo @Foo(%loopbackDouble) : (!esi.channel<i1>) -> (!esi.channel<i1>)
  %data, %valid = esi.unwrap.vr %loopback, %ready : i1
  // expected-error @+1 {{'comb.add' op operation not allowed in ESI pure modules}}
  %double = comb.add %data, %data : i1
  %loopbackDouble, %ready = esi.wrap.vr %double, %valid : i1
}

// -----

esi.pure_module @top {
  // expected-note @+1 {{}}
  %a0 = esi.pure_module.input "a" : i1
  // expected-error @+1 {{port 'a' previously declared as type 'i1'}}
  %a1 = esi.pure_module.input "a" : i5
}

// -----

esi.pure_module @top {
  // expected-note @+1 {{}}
  %a0 = esi.pure_module.input "a" : i1
  // expected-error @+1 {{port 'a' previously declared}}
  esi.pure_module.output "a", %a0 : i1
}
