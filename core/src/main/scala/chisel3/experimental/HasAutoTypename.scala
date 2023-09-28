package chisel3.experimental

import chisel3.{Data, Record}
import chisel3.internal.{sanitize, Builder}
import chisel3.util.simpleClassName

/** Trait for [[Record]]s that signals the compiler plugin to generate a typeName for the
  * inheriting [[Record]] based on the parameter values provided to its constructor.
  *
  * @example Consider a [[Bundle]] which manually implements `typeName`:
  * {{{
  * class MyBundle(param1: Int, param2: Int)(gen: Data) extends Bundle {
  *   val foo = UInt(param1.W)
  *   val bar = UInt(param2.W)
  *   val data = gen
  *   override def typeName = s"MyBundle_\${param1}_\${param2}_\${gen.typeName}"
  * }
  *
  * (new MyBundle(3, 4)(SInt(3.W))).typeName // "MyBundle_3_4_SInt3"
  * (new MyBundle(1, 32)(Bool())).typeName   // "MyBundle_1_32_Bool"
  * }}}
  *
  * An identical `typeName` implementation can be generated and provided with `HasAutoTypename`, making
  * the manual implementation unnecessary:
  * {{{
  * class MyBundle(param1: Int, param2: Int)(gen: Data) extends Bundle with HasAutoTypename {
  *   val foo = UInt(param1.W)
  *   val bar = UInt(param2.W)
  *   val data = gen
  * }
  *
  * (new MyBundle(3, 4)(SInt(3.W))).typeName // "MyBundle_3_4_SInt3"
  * (new MyBundle(1, 32)(Bool())).typeName   // "MyBundle_1_32_Bool"
  * }}}
  */
trait HasAutoTypename {
  this: Record =>

  /** Auto generate a type name for this Bundle using the bundle arguments supplied by the compiler plugin.
    */
  override def typeName: String = autoTypeName(simpleClassName(this.getClass), _typeNameConParams)

  private def autoTypeName(bundleName: String, typeNameParams: Iterable[Any]): String = {
    _typeNameConParams.foldLeft(sanitize(bundleName)) {
      case (prev, accessor) => prev + s"_${accessorString(accessor)}"
    }
  }

  private def accessorString(accessor: Any): String = accessor match {
    case d: Data => d.typeName
    case s: String =>
      sanitize(s, true) // Allow leading digits since this accessor string will occur after an underscore
    case _ => s"$accessor"
  }
}
