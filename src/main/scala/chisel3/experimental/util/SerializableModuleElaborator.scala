// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.util

import geny.Readable

import chisel3.RawModule
import chisel3.experimental.{SerializableModule, SerializableModuleGenerator, SerializableModuleParameter}

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe.{runtimeMirror, typeOf}

/** Mixin this trait to produce elaborators for [[SerializableModule]]
  */
trait SerializableModuleElaborator {

  /**
    * Implementation of a config API to serialize the [[SerializableModuleParameter]]
    * @example
    * {{{
    *  def config(parameter: MySerializableModuleParameter) =
    *    os.write.over(os.pwd / "config.json", configImpl(parameter))
    * }}}
    */
  def configImpl[P <: SerializableModuleParameter: universe.TypeTag](
    parameter: P
  )(
    implicit rwP: upickle.default.Writer[P]
  ): Readable = upickle.default.write(parameter)

  /**
    * Implementation of a design API to elaborate [[SerializableModule]]
    *
    * @return A tuple of Readable, where the first is the firrtl and the second is the serializable annotations
    * @example
    * {{{
    *  def design(parameter: os.Path) = {
    *    val (firrtl, annos) = designImpl[MySerializableModule, MySerializableModuleParameter](os.read.stream(parameter))
    *    os.write.over(os.pwd / "GCD.fir", firrtl)
    *    os.write.over(os.pwd / "GCD.anno.json", annos)
    *  }
    * }}}
    */
  def designImpl[M <: SerializableModule[P]: universe.TypeTag, P <: SerializableModuleParameter: universe.TypeTag](
    parameter: Readable
  )(
    implicit
    rwP: upickle.default.Reader[P]
  ): (Readable, Readable) = {
    var fir: firrtl.ir.Circuit = null
    val annos = Seq(
      new chisel3.stage.phases.Elaborate,
      new chisel3.stage.phases.Convert
    ).foldLeft(
      Seq(
        chisel3.stage.ChiselGeneratorAnnotation(() =>
          SerializableModuleGenerator(
            runtimeMirror(getClass.getClassLoader)
              .runtimeClass(typeOf[M].typeSymbol.asClass)
              .asInstanceOf[Class[M]],
            upickle.default.read[P](parameter)
          ).module().asInstanceOf[RawModule]
        )
      ): firrtl.AnnotationSeq
    ) { case (annos, stage) => stage.transform(annos) }
      .flatMap {
        case firrtl.stage.FirrtlCircuitAnnotation(circuit) =>
          fir = circuit
          None
        case _: firrtl.options.Unserializable => None
        case a => Some(a)
      }
    val firrtlStream: Readable = fir.serialize
    val annoStream:   Readable = firrtl.annotations.JsonProtocol.serializeRecover(annos)
    (firrtlStream, annoStream)
  }
}
