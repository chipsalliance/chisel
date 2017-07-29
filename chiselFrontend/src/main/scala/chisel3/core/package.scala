// See LICENSE for license details.

package chisel3 {
  import internal.Builder

  package object core {
    import internal.firrtl.{Width, BinaryPoint}

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
    *
    * Implementation note: the empty parameter list (like `U()`) is necessary to prevent
    * interpreting calls that have a non-Width parameter as a chained apply, otherwise things like
    * `0.asUInt(16)` (instead of `16.W`) compile without error and produce undesired results.
    */
    implicit class fromBigIntToLiteral(val bigint: BigInt) {
      /** Int to UInt conversion, recommended style for constants.
        */
      def U: UInt = UInt.Lit(bigint, Width())  // scalastyle:ignore method.name
      /** Int to SInt conversion, recommended style for constants.
        */
      def S: SInt = SInt.Lit(bigint, Width())  // scalastyle:ignore method.name
      /** Int to UInt conversion with specified width, recommended style for constants.
        */
      def U(width: Width): UInt = UInt.Lit(bigint, width)  // scalastyle:ignore method.name
      /** Int to SInt conversion with specified width, recommended style for constants.
        */
      def S(width: Width): SInt = SInt.Lit(bigint, width)  // scalastyle:ignore method.name

      /** Int to UInt conversion, recommended style for variables.
        */
      def asUInt(): UInt = UInt.Lit(bigint, Width())
      /** Int to SInt conversion, recommended style for variables.
        */
      def asSInt(): SInt = SInt.Lit(bigint, Width())
      /** Int to UInt conversion with specified width, recommended style for variables.
        */
      def asUInt(width: Width): UInt = UInt.Lit(bigint, width)
      /** Int to SInt conversion with specified width, recommended style for variables.
        */
      def asSInt(width: Width): SInt = SInt.Lit(bigint, width)
    }

    implicit class fromIntToLiteral(val int: Int) extends fromBigIntToLiteral(int)
    implicit class fromLongToLiteral(val long: Long) extends fromBigIntToLiteral(long)

    implicit class fromStringToLiteral(val str: String) {
      /** String to UInt parse, recommended style for constants.
        */
      def U: UInt = str.asUInt() // scalastyle:ignore method.name
      /** String to UInt parse with specified width, recommended style for constants.
        */
      def U(width: Width): UInt = str.asUInt(width) // scalastyle:ignore method.name

      /** String to UInt parse, recommended style for variables.
        */
      def asUInt(): UInt = {
        val bigInt = parse(str)
        UInt.Lit(bigInt, Width(bigInt.bitLength max 1))
      }
      /** String to UInt parse with specified width, recommended style for variables.
        */
      def asUInt(width: Width): UInt = UInt.Lit(parse(str), width)

      protected def parse(n: String) = {
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
    }

    implicit class fromBooleanToLiteral(val boolean: Boolean) {
      /** Boolean to Bool conversion, recommended style for constants.
        */
      def B: Bool = Bool.Lit(boolean)  // scalastyle:ignore method.name

      /** Boolean to Bool conversion, recommended style for variables.
        */
      def asBool(): Bool = Bool.Lit(boolean)
    }

    //scalastyle:off method.name
    implicit class fromDoubleToLiteral(val double: Double) {
      @deprecated("Use notation <double>.F(<binary_point>.BP) instead", "chisel3")
      def F(binaryPoint: Int): FixedPoint = FixedPoint.fromDouble(double, binaryPoint = binaryPoint)
      def F(binaryPoint: BinaryPoint): FixedPoint = {
        FixedPoint.fromDouble(double, Width(), binaryPoint)
      }
      def F(width: Width, binaryPoint: BinaryPoint): FixedPoint = {
        FixedPoint.fromDouble(double, width, binaryPoint)
      }
    }

    implicit class fromIntToWidth(val int: Int) {
      def W: Width = Width(int)  // scalastyle:ignore method.name
    }

    implicit class fromIntToBinaryPoint(val int: Int) {
      def BP: BinaryPoint = BinaryPoint(int)  // scalastyle:ignore method.name
    }
  }
}
