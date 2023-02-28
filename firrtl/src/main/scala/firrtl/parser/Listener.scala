// SPDX-License-Identifier: Apache-2.0

package firrtl.parser

import firrtl.antlr.{FIRRTLParser, _}
import firrtl.Visitor
import firrtl.Parser.InfoMode
import firrtl.ir._
import firrtl.UnsupportedVersionException

import scala.collection.mutable
import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration

private[firrtl] class Listener(infoMode: InfoMode) extends FIRRTLBaseListener {
  private var main: Option[String] = None
  private var info: Option[Info] = None
  private val modules = mutable.ArrayBuffer.empty[DefModule]

  private val visitor = new Visitor(infoMode)

  override def exitModule(ctx: FIRRTLParser.ModuleContext): Unit = {
    val m = visitor.visitModule(ctx)
    ctx.children = null // Null out to save memory
    modules += m
  }

  override def exitCircuit(ctx: FIRRTLParser.CircuitContext): Unit = {
    if (ctx.version != null) {
      val version = ctx.version.semver.getText
      val parts = version.split("\\.")
      val (major, minor, patch) = (parts(0).toInt, parts(1).toInt, parts(2).toInt)
      if (Version(major, minor, patch).incompatible(Serializer.version)) {
        throw new UnsupportedVersionException(
          s"FIRRTL version ${version} is not supported (greater than ${Serializer.version.serialize})"
        )
      }
    }
    info = Some(visitor.visitInfo(Option(ctx.info), ctx))
    main = Some(ctx.id.getText)
    ctx.children = null // Null out to save memory
  }

  def getCircuit: Circuit = {
    require(main.nonEmpty)
    val mods = modules.toSeq
    Circuit(info.get, mods, main.get)
  }
}
