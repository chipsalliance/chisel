// RUN: circt-opt %s --lower-msft-to-hw --msft-export-tcl=tops=top -verify-diagnostics -split-input-file

hw.module.extern @Foo()

hw.globalRef @ref [#hw.innerNameRef<@top::@foo1>]
// expected-error @+1 {{'msft.pd.physregion' op could not find physical region declaration named @region1}}
msft.pd.physregion @ref @region1

msft.module @top {} () -> () {
  msft.instance @foo1 @Foo() {circt.globalRef = [#hw.globalNameRef<@ref>]} : () -> ()
  msft.output
}
