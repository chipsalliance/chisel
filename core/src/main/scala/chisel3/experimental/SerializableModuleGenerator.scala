package chisel3.experimental

import upickle.default._

import scala.reflect.runtime.universe

/** Parameter for SerializableModule, it should be serializable via upickle API.
  * For more information, please refer to [[https://com-lihaoyi.github.io/upickle/]]
  */
trait SerializableModuleParameter

/** Mixin this trait to let chisel auto serialize module, it has these constraints:
  * 1. Module should not be any inner class of other class, since serializing outer class is impossible.
  * 2. Module should have and only have one parameter with type `T`:
  * {{{
  * class FooSerializableModule[FooSerializableModuleParameter](val parameter: FooSerializableModuleParameter)
  * }}}
  * 3. user should guarantee the module is reproducible on their own.
  */
trait SerializableModule[T <: SerializableModuleParameter] { this: BaseModule =>
  val parameter: T
}
object SerializableModuleGenerator {

  /** serializer for SerializableModuleGenerator. */
  implicit def rw[P <: SerializableModuleParameter, M <: SerializableModule[P]](
    implicit rwP: ReadWriter[P],
    pTypeTag:     universe.TypeTag[P],
    mTypeTag:     universe.TypeTag[M]
  ): ReadWriter[SerializableModuleGenerator[M, P]] = readwriter[ujson.Value].bimap[SerializableModuleGenerator[M, P]](
    { (x: SerializableModuleGenerator[M, P]) =>
      ujson
        .Obj(
          "parameter" -> upickle.default.writeJs[P](x.parameter),
          "generator" -> x.generator.getName
        )
    },
    { (json: ujson.Value) =>
      SerializableModuleGenerator[M, P](
        Class.forName(json.obj("generator").str).asInstanceOf[Class[M]],
        upickle.default.read[P](json.obj("parameter"))
      )
    }
  )
}

/** the serializable module generator:
  * @param generator a non-inner class of module, which should be a subclass of [[SerializableModule]]
  * @param parameter the parameter of `generator`
  */
case class SerializableModuleGenerator[M <: SerializableModule[P], P <: SerializableModuleParameter](
  generator: Class[M],
  parameter: P
)(
  implicit val pTag: universe.TypeTag[P],
  implicit val mTag: universe.TypeTag[M]) {
  private[chisel3] def construct: M = {
    require(
      generator.getConstructors.length == 1,
      s"""only allow constructing SerializableModule from SerializableModuleParameter via class Module(val parameter: Parameter),
         |you have ${generator.getConstructors.length} constructors
         |""".stripMargin
    )
    require(
      !generator.getConstructors.head.getParameterTypes.last.toString.contains("$"),
      s"""You define your ${generator.getConstructors.head.getParameterTypes.last} inside other class.
         |This is a forbidden behavior, since we cannot serialize the out classes,
         |for debugging, these are full parameter types of constructor:
         |${generator.getConstructors.head.getParameterTypes.mkString("\n")}
         |""".stripMargin
    )
    require(
      generator.getConstructors.head.getParameterCount == 1,
      s"""You define multiple constructors:
         |${generator.getConstructors.head.getParameterTypes.mkString("\n")}
         |""".stripMargin
    )
    generator.getConstructors.head.newInstance(parameter).asInstanceOf[M]
  }

  /** elaborate a module from this generator. */
  def module(): M = construct
}
