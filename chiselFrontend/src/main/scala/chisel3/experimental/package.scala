// See LICENSE for license details.

package chisel3

/** Package for experimental features, which may have their API changed, be removed, etc.
  *
  * Because its contents won't necessarily have the same level of stability and support as
  * non-experimental, you must explicitly import this package to use its contents.
  */
package object experimental {  // scalastyle:ignore object.name

  // BlackBox Parameters
  type Param = chisel3.Param
  type IntParam = chisel3.IntParam
  val IntParam = chisel3.IntParam
  type DoubleParam = chisel3.DoubleParam
  val DoubleParam = chisel3.DoubleParam
  type StringParam = chisel3.StringParam
  val StringParam = chisel3.StringParam
  type RawParam = chisel3.RawParam
  val RawParam = chisel3.RawParam

  // Implicit conversions for BlackBox Parameters
  implicit def fromIntToIntParam(x: Int): IntParam = IntParam(BigInt(x))
  implicit def fromLongToIntParam(x: Long): IntParam = IntParam(BigInt(x))
  implicit def fromBigIntToIntParam(x: BigInt): IntParam = IntParam(x)
  implicit def fromDoubleToDoubleParam(x: Double): DoubleParam = DoubleParam(x)
  implicit def fromStringToStringParam(x: String): StringParam = StringParam(x)

  type Analog = chisel3.Analog
  val Analog = chisel3.Analog

  type ChiselEnum = EnumFactory

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
    def apply(proto: BaseModule)(implicit sourceInfo: internal.sourceinfo.SourceInfo, compileOptions: CompileOptions): ClonePorts = { // scalastyle:ignore line.size.limit
      BaseModule.cloneIORecord(proto)
    }
  }

  val requireIsHardware = core.requireIsHardware
  val requireIsChiselType = core.requireIsChiselType
  type Direction = ActualDirection
  val Direction = ActualDirection

  implicit class ChiselRange(val sc: StringContext) extends AnyVal {
    import internal.firrtl.NumericBound

    import scala.language.experimental.macros

    /** Specifies a range using mathematical range notation. Variables can be interpolated using
      * standard string interpolation syntax.
      * @example {{{
      * UInt(range"[0, 2)")
      * UInt(range"[0, \$myInt)")
      * UInt(range"[0, \${myInt + 2})")
      * }}}
      */
    def range(args: Any*): (NumericBound[Int], NumericBound[Int]) = macro internal.RangeTransform.apply
  }

  class dump extends internal.naming.dump  // scalastyle:ignore class.name
  class treedump extends internal.naming.treedump  // scalastyle:ignore class.name
  class chiselName extends internal.naming.chiselName  // scalastyle:ignore class.name
}
