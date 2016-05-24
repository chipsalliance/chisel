package object Chisel {
  /**
  * These implicit classes allow one to convert scala.Int|scala.BigInt to
  * Chisel.UInt|Chisel.SInt by calling .asUInt|.asSInt on them, respectively.
  * The versions .asUInt(width)|.asSInt(width) are also available to explicitly
  * mark a width for the new literal.
  *
  * Also provides .asBool to scala.Boolean and .asUInt to String
  *
  * Note that, for stylistic reasons, one hould avoid extracting immediately
  * after this call using apply, ie. 0.asUInt(1)(0) due to potential for
  * confusion (the 1 is a bit length and the 0 is a bit extraction position).
  * Prefer storing the result and then extracting from it.
  */
  implicit class addLiteraltoScalaInt(val target: Int) extends AnyVal {
    def asUInt() = UInt.Lit(target)
    def asSInt() = SInt.Lit(target)
    def asUInt(width: Int) = UInt.Lit(target, width)
    def asSInt(width: Int) = SInt.Lit(target, width)

    // These were recently added to chisel2/3 but are not to be used internally
    @deprecated("asUInt should be used over U", "gchisel")
    def U() = UInt.Lit(target)
    @deprecated("asSInt should be used over S", "gchisel")
    def S() = SInt.Lit(target)
    @deprecated("asUInt should be used over U", "gchisel")
    def U(width: Int) = UInt.Lit(target, width)
    @deprecated("asSInt should be used over S", "gchisel")
    def S(width: Int) = SInt.Lit(target, width)
  }
  implicit class addLiteraltoScalaBigInt(val target: BigInt) extends AnyVal {
    def asUInt() = UInt.Lit(target)
    def asSInt() = SInt.Lit(target)
    def asUInt(width: Int) = UInt.Lit(target, width)
    def asSInt(width: Int) = SInt.Lit(target, width)

    // These were recently added to chisel2/3 but are not to be used internally
    @deprecated("asUInt should be used over U", "gchisel")
    def U() = UInt.Lit(target)
    @deprecated("asSInt should be used over S", "gchisel")
    def S() = SInt.Lit(target)
    @deprecated("asUInt should be used over U", "gchisel")
    def U(width: Int) = UInt.Lit(target, width)
    @deprecated("asSInt should be used over S", "gchisel")
    def S(width: Int) = SInt.Lit(target, width)
  }
  implicit class addLiteraltoScalaString(val target: String) extends AnyVal {
    def asUInt() = UInt.Lit(target)
    def asUInt(width: Int) = UInt.Lit(target, width)

    // These were recently added to chisel2/3 but are not to be used internally
    @deprecated("asUInt should be used over U", "gchisel")
    def U() = UInt.Lit(target)
    @deprecated("asUInt should be used over U", "gchisel")
    def U(width: Int) = UInt.Lit(target, width)
  }
  implicit class addLiteraltoScalaBool(val target: Boolean) extends AnyVal {
    def asBool = Bool.Lit(target)

    // These were recently added to chisel2/3 but are not to be used internally
    @deprecated("asBool should be used over B", "gchisel")
    def B = Bool.Lit(target)
  }
}
