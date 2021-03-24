// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.sourceinfo.SourceInfo

/** Package for experimental features, which may have their API changed, be removed, etc.
  *
  * Because its contents won't necessarily have the same level of stability and support as
  * non-experimental, you must explicitly import this package to use its contents.
  */
package object experimental {
  import scala.language.implicitConversions
  import chisel3.internal.BaseModule

  // Implicit conversions for BlackBox Parameters
  implicit def fromIntToIntParam(x: Int): IntParam = IntParam(BigInt(x))
  implicit def fromLongToIntParam(x: Long): IntParam = IntParam(BigInt(x))
  implicit def fromBigIntToIntParam(x: BigInt): IntParam = IntParam(x)
  implicit def fromDoubleToDoubleParam(x: Double): DoubleParam = DoubleParam(x)
  implicit def fromStringToStringParam(x: String): StringParam = StringParam(x)

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
    def apply(proto: BaseModule)(implicit sourceInfo: chisel3.internal.sourceinfo.SourceInfo, compileOptions: CompileOptions): ClonePorts = {
      BaseModule.cloneIORecord(proto)
    }
  }

  val requireIsHardware = chisel3.internal.requireIsHardware
  val requireIsChiselType =  chisel3.internal.requireIsChiselType
  type Direction = ActualDirection
  val Direction = ActualDirection

  implicit class ChiselRange(val sc: StringContext) extends AnyVal {

    import scala.language.experimental.macros

    /** Specifies a range using mathematical range notation. Variables can be interpolated using
      * standard string interpolation syntax.
      * @example {{{
      * UInt(range"[0, 2)")
      * UInt(range"[0, \$myInt)")
      * UInt(range"[0, \${myInt + 2})")
      * }}}
      */
    def range(args: Any*): chisel3.internal.firrtl.IntervalRange = macro chisel3.internal.RangeTransform.apply
  }

  class dump extends chisel3.internal.naming.dump
  class treedump extends chisel3.internal.naming.treedump
  /** Experimental macro for naming Chisel hardware values
    *
    * By default, Chisel uses reflection for naming which only works for public fields of `Bundle`
    * and `Module` classes. Applying this macro annotation to a `class` or `object` enables Chisel
    * to name any hardware values within the annotated `class` or `object.
    *
    * @example {{{
    * import chisel3._
    * import chisel3.experimental.chiselName
    *
    * @chiselName
    * class MyModule extends Module {
    *   val io = IO(new Bundle {
    *     val in = Input(UInt(8.W))
    *     val out = Output(UInt(8.W))
    *   })
    *   def createReg(): Unit = {
    *     // @chiselName allows Chisel to name this Reg
    *     val myReg = RegInit(io.in)
    *     io.out := myReg
    *   }
    *   createReg()
    * }
    * }}}
    */
  class chiselName extends chisel3.internal.naming.chiselName
  /** Do not name instances of this type in [[chiselName]]
    *
    * By default, `chiselName` will include `val` names of instances of annotated classes as a
    * prefix in final naming. Mixing in this trait to a `class`, `object`, or anonymous `class`
    * instances will exclude the `val` name from `chiselName` naming.
    *
    * @example {{{
    * import chisel3._
    * import chisel3.experimental.{chiselName, NoChiselNamePrefix}
    *
    * // Note that this is not a Module
    * @chiselName
    * class Counter(w: Int) {
    *   val myReg = RegInit(0.U(w.W))
    *   myReg := myReg + 1.U
    * }
    *
    * @chiselName
    * class MyModule extends Module {
    *   val io = IO(new Bundle {
    *     val out = Output(UInt(8.W))
    *   })
    *   // Name of myReg will be "counter0_myReg"
    *   val counter0 = new Counter(8)
    *   // Name of myReg will be "myReg"
    *   val counter1 = new Counter(8) with NoChiselNamePrefix
    *   io.out := counter0.myReg + counter1.myReg
    * }
    * }}}
    */
  trait NoChiselNamePrefix

  object BundleLiterals {
    implicit class AddBundleLiteralConstructor[T <: Record](x: T) {
      def Lit(elems: (T => (Data, Data))*): T = {
        x._makeLit(elems: _*)
      }
    }
  }

  /** This class provides the `Lit` method needed to define a `Vec` literal
    */
  object VecLiterals {
    implicit class AddVecLiteralConstructor[T <: Vec[_]](x: T) {
      /** Given a generator of a list tuples of the form [Int, Data]
        * constructs a Vec literal, parallel concept to `BundleLiteral`
        *
        * @param elems generates a lit of `(Int, Data)` where the I
        * @return
        */
      def Lit(elems: (Int, Data)*): T = {
        x._makeLit(elems: _*)
      }
    }

    implicit class AddObjectLiteralConstructor[T <: Vec.type](x: T) {
      def Lit[T <: Data](elems: T*)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] = {
        require(elems.nonEmpty, s"Lit.Vec(...) must have at least one element")
        val indexElements = elems.zipWithIndex.map { case (element, index) => (index, element)}
        val widestElement = elems.sortBy(_.getWidth).last
        val vec: Vec[T] = Vec.apply(indexElements.length, chiselTypeOf(widestElement))
        vec.Lit(indexElements:_*)
      }
    }
  }

  // Use to add a prefix to any component generated in input scope
  val prefix = chisel3.internal.prefix
  // Use to remove prefixes not in provided scope
  val noPrefix = chisel3.internal.noPrefix
}
