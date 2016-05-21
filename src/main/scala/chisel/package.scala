package object chisel {
  import scala.language.experimental.macros

  import internal.firrtl.Width
  import internal.sourceinfo.{SourceInfo, SourceInfoTransform}

  implicit class fromBigIntToLiteral(val x: BigInt) extends AnyVal {
    def U: UInt = UInt(x, Width())
    def S: SInt = SInt(x, Width())
  }
  implicit class fromIntToLiteral(val x: Int) extends AnyVal {
    def U: UInt = UInt(BigInt(x), Width())
    def S: SInt = SInt(BigInt(x), Width())
  }
  implicit class fromStringToLiteral(val x: String) extends AnyVal {
    def U: UInt = UInt(x)
  }
  implicit class fromBooleanToLiteral(val x: Boolean) extends AnyVal {
    def B: Bool = Bool(x)
  }

  implicit class fromUIntToBitPatComparable(val x: UInt) extends AnyVal {
    final def === (that: BitPat): Bool = macro SourceInfoTransform.thatArg
    final def != (that: BitPat): Bool = macro SourceInfoTransform.thatArg
    final def =/= (that: BitPat): Bool = macro SourceInfoTransform.thatArg

    def do_=== (that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that === x
    def do_!= (that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that != x
    def do_=/= (that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that =/= x
  }
}
