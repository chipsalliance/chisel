package chisel3

import chisel3.domain
import chisel3.internal.{throwException, Builder}
import chisel3.experimental.{noPrefix, requireIsChiselType, SourceInfo}
import chisel3.properties.{Class, Property}
import chisel3.reflect.DataMirror.internal.chiselTypeClone
import chisel3.reflect.DataMirror.{hasProbeTypeModifier, specifiedDirectionOf}

object IO extends IO$Intf {

  private[chisel3] def _applyImpl[T <: Data](iodef: => T)(implicit sourceInfo: SourceInfo): T = {
    val module = Module.currentModule.get // Impossible to fail
    if (!module.isIOCreationAllowed)
      Builder.error(
        s"This module cannot have IOs instantiated after disallowing IOs: ${module._whereIOCreationIsDisallowed
            .map(_.makeMessage { (s: String) => s })
            .mkString(",")}"
      )
    require(!module.isClosed, "Can't add more ports after module close")
    val prevId = Builder.idGen.value
    val data = iodef // evaluate once (passed by name)
    requireIsChiselType(data, "io type")

    // Fail if the module is a Class, and the type is Data.
    module match {
      case _: Class => {
        data match {
          case _: Property[_] => ()
          case _ => Builder.error(s"Class ports must be Property type, but found ${data._localErrorContext}.")
        }
      }
      case _ => ()
    }

    // Clone the IO so we preserve immutability of data types
    // Note: we don't clone if the data is fresh (to avoid unnecessary clones)
    val iodefClone =
      if (!data.mustClone(prevId)) data
      else
        try {
          data.cloneTypeFull
        } catch {
          // For now this is going to be just a deprecation so we don't suddenly break everyone's code
          case e: AutoClonetypeException =>
            Builder.deprecated(e.getMessage, Some(s"${data.getClass}"))
            data
        }
    module.bindIoInPlace(iodefClone)
    iodefClone
  }
}

/** The same as [[IO]] except there is no prefix when given a [[Record]] or
  * [[Bundle]].  For [[Element]] ([[UInt]], etc.) or [[Vec]] types, this is
  * the same as [[IO]]. It is also the same as [[IO]] for [[chisel3.probe.Probe]] types.
  *
  * @example {{{
  * class MyBundle extends Bundle {
  *   val foo = Input(UInt(8.W))
  *   val bar = Output(UInt(8.W))
  * }
  * class MyModule extends Module {
  *   val io = FlatIO(new MyBundle)
  *   // input  [7:0] foo,
  *   // output [7:0] bar
  * }
  * }}}
  */
object FlatIO extends FlatIO$Intf {
  private[chisel3] def _applyImpl[T <: Data](gen: => T)(implicit sourceInfo: SourceInfo): T = noPrefix {
    import chisel3.experimental.dataview._

    def coerceDirection(d: Data): Data = {
      import chisel3.{SpecifiedDirection => SD}
      specifiedDirectionOf(gen) match {
        case SD.Flip   => Flipped(d)
        case SD.Input  => Input(d)
        case SD.Output => Output(d)
        case _         => d
      }
    }

    type R = T with Record
    gen match {
      case d if hasProbeTypeModifier(d) => IO(d)
      case _:      Element => IO(gen)
      case _:      Vec[_]  => IO(gen)
      case record: R =>
        val ports: Seq[Data] =
          record._elements.toSeq.reverse.map { case (name, data) =>
            val p = IO(coerceDirection(data).asInstanceOf[Data])
            p.suggestName(name)
            p

          }

        implicit val dv: DataView[Seq[Data], R] = DataView.mapping(
          _ => chiselTypeClone(gen).asInstanceOf[R],
          (seq, rec) => seq.zip(rec._elements.toSeq.reverse).map { case (port, (_, field)) => port -> field }
        )
        ports.viewAs[R]
    }
  }
}
