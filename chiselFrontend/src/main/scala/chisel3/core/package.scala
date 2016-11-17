package chisel3 {
  package object core {
    import internal.firrtl.Width

    /**
    * These implicit classes allow one to convert scala.Int|scala.BigInt to
    * Chisel.UInt|Chisel.SInt by calling .asUInt|.asSInt on them, respectively.
    * The versions .asUInt(width)|.asSInt(width) are also available to explicitly
    * mark a width for the new literal.
    *
    * Also provides .asBool to scala.Boolean and .asUInt to String
    *
    * Note that, for stylistic reasons, one should avoid extracting immediately
    * after this call using apply, ie. 0.asUInt(1)(0) due to potential for
    * confusion (the 1 is a bit length and the 0 is a bit extraction position).
    * Prefer storing the result and then extracting from it.
    */
    implicit class fromIntToLiteral(val x: Int) {
      def U: UInt = UInt.Lit(BigInt(x), Width())    // scalastyle:ignore method.name
      def S: SInt = SInt(BigInt(x), Width())    // scalastyle:ignore method.name

      def asUInt(): UInt = UInt.Lit(x, Width())
      def asSInt(): SInt = SInt(x, Width())
      def asUInt(width: Int): UInt = UInt.Lit(x, width)
      def asSInt(width: Int): SInt = SInt(x, width)
    }

    implicit class fromBigIntToLiteral(val x: BigInt) {
      def U: UInt = UInt.Lit(x, Width())       // scalastyle:ignore method.name
      def S: SInt = SInt(x, Width())       // scalastyle:ignore method.name

      def asUInt(): UInt = UInt.Lit(x, Width())
      def asSInt(): SInt = SInt(x, Width())
      def asUInt(width: Int): UInt = UInt.Lit(x, width)
      def asSInt(width: Int): SInt = SInt(x, width)
    }

    implicit class fromStringToLiteral(val x: String) {
      def U: UInt = UInt.Lit(x)       // scalastyle:ignore method.name
    }

    implicit class fromBooleanToLiteral(val x: Boolean) {
      def B: Bool = Bool(x)       // scalastyle:ignore method.name
    }

    implicit class fromDoubleToLiteral(val x: Double) {
      def F(binaryPoint: Int): FixedPoint = FixedPoint.fromDouble(x, binaryPoint = binaryPoint)
    }
  }
}