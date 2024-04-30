// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.tools.nsc
import scala.tools.nsc.plugins.PluginComponent
import scala.tools.nsc.transform.TypingTransformers
import scala.tools.nsc.{Global, Phase}

class SerializableGenerator(val global: Global)
    extends PluginComponent
    with TypingTransformers {
  import global._
  val runsAfter: List[String] = List[String]("typer")
  val phaseName = "serializablemodulephase"
  def newPhase(_prev: Phase): SerializableModulePhase = new SerializableModulePhase(_prev)
  class SerializableModulePhase(prev: Phase) extends StdPhase(prev) {
    override def name: String = phaseName
    def apply(unit: CompilationUnit): Unit = {
        unit.body = {
          new TypingTransformer(unit) {
            def inferType(t: Tree): Type = localTyper.typed(t, nsc.Mode.TYPEmode).tpe
            override def transform(tree: Tree): Tree = {
              tree match {
                case cd: ClassDef
                  if
                    // Module should be concrete
                    !cd.symbol.hasAbstractFlag &&
                    // Module should extend from SerializableModule
                    cd.impl.parents.exists {
                      case a if inferType(a) <:< inferType(tq"chisel3.experimental.SerializableModule[_]") => true
                      case _ => false
                    } =>

                  val moduleCompanion = cd.symbol.companionModule
                  val moduleTypeName = cd.name
                  val moduleCompanionName = moduleTypeName.companionName

                  val parameter = getSerializableModuleParameter(cd)
                  val parameterCompanion = parameter.companionName
                  val parameterTermName = getSerializableModuleParameter(cd).toTermName
                  val parameterTypeName: global.TypeName = parameterTermName.toTypeName
                  val companion =
                    q"""
                  object $moduleCompanionName extends chisel3.experimental.SerializableModuleMain[$parameterTypeName, $moduleTypeName] {
                    import scala.reflect.runtime.universe
                    implicit val pRW: default.ReadWriter[$parameterTypeName] = $parameterTermName.rw
                    implicit val mTypeTag: universe.TypeTag[$moduleTypeName] = implicitly[universe.TypeTag[$moduleTypeName]]
                    implicit val pTypeTag: universe.TypeTag[$parameterTypeName] = implicitly[universe.TypeTag[$parameterTypeName]]
                  }"""
                  println(companion)
                  super.transform(tree)
                case _ => super.transform(tree)
              }
            }
            // Super dirty...
            private def getSerializableModuleParameter(cd: ClassDef): global.Name = {
              cd.impl.parents.collectFirst {
                case tt: TypeTree => tt.original match {
                  case att: AppliedTypeTree => att.args match {
                    case List(p) => p match {
                      case tt: TypeTree => tt.original.asInstanceOf[NameTreeApi].name
                    }
                  }
                }
              }.get
            }
          }.transform(unit.body)
          unit.body
        }
      }
    }
}
