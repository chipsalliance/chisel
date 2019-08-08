// See LICENSE for license details.

package chisel3

/** Package for experimental features, which may have their API changed, be removed, etc.
  *
  * Because its contents won't necessarily have the same level of stability and support as
  * non-experimental, you must explicitly import this package to use its contents.
  */
package object experimental {  // scalastyle:ignore object.name
  import scala.language.implicitConversions
  import chisel3.internal.BaseModule

  // Implicit conversions for BlackBox Parameters
  implicit def fromIntToIntParam(x: Int): IntParam = IntParam(BigInt(x))
  implicit def fromLongToIntParam(x: Long): IntParam = IntParam(BigInt(x))
  implicit def fromBigIntToIntParam(x: BigInt): IntParam = IntParam(x)
  implicit def fromDoubleToDoubleParam(x: Double): DoubleParam = DoubleParam(x)
  implicit def fromStringToStringParam(x: String): StringParam = StringParam(x)

  type ChiselEnum = EnumFactory

  @deprecated("Use the version in chisel3._", "3.2")
  val withClockAndReset = chisel3.withClockAndReset
  @deprecated("Use the version in chisel3._", "3.2")
  val withClock = chisel3.withClock
  @deprecated("Use the version in chisel3._", "3.2")
  val withReset = chisel3.withReset

  // Rocket Chip-style clonemodule

  /** A record containing the results of CloneModuleAsRecord
    * The apply method is retrieves the element with the supplied name.
    */
  type ClonePorts = BaseModule.ClonePorts

  object CloneModuleAsRecord {
    /** Clones an existing module and returns a record of all its top-level ports.
      * Each element of the record is named with a string matching the
      * corresponding port's name and shares the port's type.
      * @example {{{
      * val q1 = Module(new Queue(UInt(32.W), 2))
      * val q2_io = CloneModuleAsRecord(q1)("io").asInstanceOf[q1.io.type]
      * q2_io.enq <> q1.io.deq
      * }}}
      */
    def apply(proto: BaseModule)(implicit sourceInfo: chisel3.internal.sourceinfo.SourceInfo, compileOptions: CompileOptions): ClonePorts = { // scalastyle:ignore line.size.limit
      BaseModule.cloneIORecord(proto)
    }
  }

  val requireIsHardware = chisel3.internal.requireIsHardware
  val requireIsChiselType =  chisel3.internal.requireIsChiselType
  type Direction = ActualDirection
  val Direction = ActualDirection

  implicit class ChiselRange(val sc: StringContext) extends AnyVal {
    import chisel3.internal.firrtl.NumericBound

    import scala.language.experimental.macros

    /** Specifies a range using mathematical range notation. Variables can be interpolated using
      * standard string interpolation syntax.
      * @example {{{
      * UInt(range"[0, 2)")
      * UInt(range"[0, \$myInt)")
      * UInt(range"[0, \${myInt + 2})")
      * }}}
      */
    def range(args: Any*): (NumericBound[Int], NumericBound[Int]) = macro chisel3.internal.RangeTransform.apply
  }

  class dump extends chisel3.internal.naming.dump  // scalastyle:ignore class.name
  class treedump extends chisel3.internal.naming.treedump  // scalastyle:ignore class.name
  class chiselName extends chisel3.internal.naming.chiselName  // scalastyle:ignore class.name

  object BundleLiterals {
    implicit class AddBundleLiteralConstructor[T <: Bundle](x: T) {
      def Lit(elems: (T => (Data, Data))*): T = {
        x._makeLit(elems: _*)
      }
    }
  }
}
