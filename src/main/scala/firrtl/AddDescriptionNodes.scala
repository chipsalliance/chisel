// See LICENSE for license details.

package firrtl

import firrtl.ir._
import firrtl.annotations._
import firrtl.Mappers._

case class DescriptionAnnotation(named: Named, description: String) extends Annotation {
  def update(renames: RenameMap): Seq[DescriptionAnnotation] = {
    renames.get(named) match {
      case None => Seq(this)
      case Some(seq) => seq.map(n => this.copy(named = n))
    }
  }
}

private sealed trait HasDescription {
  def description: Description
}

private abstract class Description extends FirrtlNode

private case class DocString(string: StringLit) extends Description {
  def serialize: String = "@[" + string.serialize + "]"
}

private case object EmptyDescription extends Description {
  def serialize: String = ""
}

private case class DescribedStmt(description: Description, stmt: Statement) extends Statement with HasDescription {
  def serialize: String = s"${description.serialize}\n${stmt.serialize}"
  def mapStmt(f: Statement => Statement): Statement = f(stmt)
  def mapExpr(f: Expression => Expression): Statement = this.copy(stmt = stmt.mapExpr(f))
  def mapType(f: Type => Type): Statement = this.copy(stmt = stmt.mapType(f))
  def mapString(f: String => String): Statement = this.copy(stmt = stmt.mapString(f))
  def mapInfo(f: Info => Info): Statement = this.copy(stmt = stmt.mapInfo(f))
}

private case class DescribedMod(description: Description,
  portDescriptions: Map[String, Description],
  mod: DefModule) extends DefModule with HasDescription {
  val info = mod.info
  val name = mod.name
  val ports = mod.ports
  def serialize: String = s"${description.serialize}\n${mod.serialize}"
  def mapStmt(f: Statement => Statement): DefModule = this.copy(mod = mod.mapStmt(f))
  def mapPort(f: Port => Port): DefModule = this.copy(mod = mod.mapPort(f))
  def mapString(f: String => String): DefModule = this.copy(mod = mod.mapString(f))
  def mapInfo(f: Info => Info): DefModule = this.copy(mod = mod.mapInfo(f))
}

/** Wraps modules or statements with their respective described nodes.
  * Descriptions come from [[DescriptionAnnotation]]. Describing a
  * module or any of its ports will turn it into a [[DescribedMod]].
  * Describing a Statement will turn it into a [[DescribedStmt]]
  *
  * @note should only be used by VerilogEmitter, described nodes will
  *       break other transforms.
  */
class AddDescriptionNodes extends Transform {
  def inputForm = LowForm
  def outputForm = LowForm

  def onStmt(compMap: Map[String, Seq[String]])(stmt: Statement): Statement = {
    stmt.map(onStmt(compMap)) match {
      case d: IsDeclaration if compMap.contains(d.name) =>
        DescribedStmt(DocString(StringLit.unescape(compMap(d.name).mkString("\n\n"))), d)
      case other => other
    }
  }

  def onModule(modMap: Map[String, Seq[String]], compMaps: Map[String, Map[String, Seq[String]]])
    (mod: DefModule): DefModule = {
    val (newMod, portDesc: Map[String, Description]) = compMaps.get(mod.name) match {
      case None => (mod, Map.empty)
      case Some(compMap) => (mod.mapStmt(onStmt(compMap)), mod.ports.collect {
        case p @ Port(_, name, _, _) if compMap.contains(name) =>
          name -> DocString(StringLit.unescape(compMap(name).mkString("\n\n")))
      }.toMap)
    }

    val modDesc = modMap.get(newMod.name).map {
      desc => DocString(StringLit.unescape(desc.mkString("\n\n")))
    }

    if (portDesc.nonEmpty || modDesc.nonEmpty) {
      DescribedMod(modDesc.getOrElse(EmptyDescription), portDesc, newMod)
    } else {
      newMod
    }
  }

  def collectMaps(annos: Seq[Annotation]): (Map[String, Seq[String]], Map[String, Map[String, Seq[String]]]) = {
    val modMap = annos.collect {
      case DescriptionAnnotation(ModuleName(m, CircuitName(c)), desc) => (m, desc)
    }.groupBy(_._1).mapValues(_.map(_._2))

    val compMap = annos.collect {
      case DescriptionAnnotation(ComponentName(comp, ModuleName(mod, CircuitName(circ))), desc) =>
        (mod, comp, desc)
    }.groupBy(_._1).mapValues(_.groupBy(_._2).mapValues(_.map(_._3)))

    (modMap, compMap)
  }

  def executeModule(module: DefModule, annos: Seq[Annotation]): DefModule = {
    val (modMap, compMap) = collectMaps(annos)

    onModule(modMap, compMap)(module)
  }

  override def execute(state: CircuitState): CircuitState = {
    val (modMap, compMap) = collectMaps(state.annotations)

    state.copy(circuit = state.circuit.mapModule(onModule(modMap, compMap)))
  }
}
