hw.module.extern @ichi(%a: i2, %b: i3) -> (%c: i4, %d: i5)

hw.module @owo() -> (%owo_result : i32) {
  %0 = hw.constant 3 : i32
  hw.output %0 : i32
}
hw.module @uwu() -> () { }

hw.module @nya(%nya_input : i32) -> () {
  hw.instance "uwu1" @uwu() : () -> ()
}

hw.module @test() -> (%test_result : i32) {
  %myArray1 = sv.wire : !hw.inout<array<42 x i8>>
  %0 = hw.instance "owo1" @owo() : () -> (i32)
  hw.instance "nya1" @nya(%0) : (i32) -> ()
  hw.output %0 : i32
}

hw.module @always() -> () {
  %clock = hw.constant 1 : i1
  %3 = sv.wire : !hw.inout<i1>
  %false = hw.constant 0 : i1
  sv.alwaysff(posedge %clock)  {
    sv.passign %3, %false : i1
  }
  hw.output
}

