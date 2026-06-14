// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.util

import geny.Readable
import chisel3.RawModule
import chisel3.experimental.{SerializableModule, SerializableModuleGenerator, SerializableModuleParameter}
import firrtl.annotations.Annotation

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe.{runtimeMirror, typeOf}

/** Mixin this trait to produce elaborators for [[SerializableModule]]
  */
trait SerializableModuleElaborator {
  def additionalAnnotations: Seq[Annotation] = Nil

  /**
    * Implementation of a config API to serialize the [[SerializableModuleParameter]]
    * @example
    * {{{
    *  def config(parameter: MySerializableModuleParameter): Unit = {
    *    val out = java.nio.file.Files.newOutputStream(java.nio.file.Paths.get("config.json"))
    *    try configImpl(parameter).writeBytesTo(out)
    *    finally out.close()
    *  }
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
    *  def design(parameter: java.nio.file.Path): Unit = {
    *    val input = new String(java.nio.file.Files.readAllBytes(parameter), java.nio.charset.StandardCharsets.UTF_8)
    *    val (firrtl, annos) = designImpl[MySerializableModule, MySerializableModuleParameter](input)
    *    writeReadable(firrtl, java.nio.file.Paths.get("GCD.fir"))
    *    writeReadable(annos, java.nio.file.Paths.get("GCD.anno.json"))
    *  }
    *  def writeReadable(data: geny.Readable, path: java.nio.file.Path): Unit = {
    *    val out = java.nio.file.Files.newOutputStream(path)
    *    try data.writeBytesTo(out)
    *    finally out.close()
    *  }
    * }}}
    */
  def designImpl[M <: SerializableModule[P]: universe.TypeTag, P <: SerializableModuleParameter: universe.TypeTag](
    parameter: Readable
  )(
    implicit rwP: upickle.default.Reader[P]
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
      ) ++ additionalAnnotations
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
