// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.reflect.internal.Flags
import scala.tools.nsc
import scala.tools.nsc.{Global, Phase}
import scala.tools.nsc.plugins.PluginComponent
import scala.tools.nsc.transform.{Transform, TypingTransformers}

class SerializableGeneratorComponent(val global: Global, arguments: ChiselPluginArguments)
  extends PluginComponent
    with TypingTransformers
    with Transform
    with ChiselOuterUtils {
  import global._
  import global.definitions._

  override val phaseName: String = "serializablegeneratorcomponent"
  override val runsAfter: List[String] = "typer" :: Nil

  override def newTransformer(unit: global.CompilationUnit): global.Transformer = new MyTypingTransformer(unit)

  private class MyTypingTransformer(unit: CompilationUnit) extends TypingTransformer(unit) with ChiselInnerUtils {
    override def transform(tree: Tree): Tree = tree match {
      // find all SerializableModule and generate corresponding SerializableGenerator in SerializableGenerator._
      case module: ClassDef
        if isASerializableModuleTpe(module.symbol) &&
          !module.mods.hasFlag(Flag.ABSTRACT) => {
        require(module.symbol.ownerChain.drop(1).forall(_.hasPackageFlag), s"${module.symbol} at ${module.pos} should only declared in packages, rather than classes.")
        val parameterTpe: global.Symbol = module.symbol.info.members.collectFirst {case member if member.name.toString == "SerializableModuleParameter" => member}.get
        println(s"detected ${module.symbol.fullName} with ${parameterTpe.info} as its parameter.")
        // create class and append to tree:
        // class "${module.symbol.name}Generator$Auto"(val parameter: ${parameterTpe.info})(implicit val parameterRW: upickle.default.ReadWriter[${parameterTpe.info}])
        //   extends chisel3.experimental.SerializableModuleGenerator {
        //   override type M = ${module.symbol.fullName}
        //   override val moduleClass = classOf[M]
        // }
        tree
      }
      case _ => super.transform(tree)
    }
  }
}