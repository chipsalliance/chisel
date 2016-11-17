package chisel3 {
  import internal.Builder

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
      def asUInt(width: Int): UInt = UInt.Lit(x, Width(width))
      def asSInt(width: Int): SInt = SInt(x, Width(width))
    }

    implicit class fromBigIntToLiteral(val x: BigInt) {
      def U: UInt = UInt.Lit(x, Width())       // scalastyle:ignore method.name
      def S: SInt = SInt(x, Width())       // scalastyle:ignore method.name

      def asUInt(): UInt = UInt.Lit(x, Width())
      def asSInt(): SInt = SInt(x, Width())
      def asUInt(width: Int): UInt = UInt.Lit(x, Width(width))
      def asSInt(width: Int): SInt = SInt(x, width)
    }

    implicit class fromStringToLiteral(val x: String) {
      def U: UInt = UInt.Lit(fromStringToLiteral.parse(x), fromStringToLiteral.parsedWidth(x))       // scalastyle:ignore method.name
    }

    object fromStringToLiteral {
      def parse(n: String) = {
        val (base, num) = n.splitAt(1)
        val radix = base match {
          case "x" | "h" => 16
          case "d" => 10
          case "o" => 8
          case "b" => 2
          case _ => Builder.error(s"Invalid base $base"); 2
        }
        BigInt(num.filterNot(_ == '_'), radix)
      }

      def parsedWidth(n: String) =
        if (n(0) == 'b') {
          Width(n.length-1)
        } else if (n(0) == 'h') {
          Width((n.length-1) * 4)
        } else {
          Width()
        }
    }

    implicit class fromBooleanToLiteral(val x: Boolean) {
      def B: Bool = Bool(x)       // scalastyle:ignore method.name
    }

    implicit class fromDoubleToLiteral(val x: Double) {
      def F(binaryPoint: Int): FixedPoint = FixedPoint.fromDouble(x, binaryPoint = binaryPoint)
    }
  }
}