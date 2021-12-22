// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.ir._
import firrtl.annotations._
import firrtl.Mappers._
import firrtl.options.Dependency

/**
  * A base trait for `Annotation`s that describe a `FirrtlNode`.
  * Usually, we would like to emit these descriptions in some way.
  */
sealed trait DescriptionAnnotation extends Annotation {
  def target:      Target
  def description: String
}

/**
  * A docstring description (a comment).
  * @param target the object being described
  * @param description the docstring describing the object
  */
case class DocStringAnnotation(target: Target, description: String) extends DescriptionAnnotation {
  def update(renames: RenameMap): Seq[DocStringAnnotation] = {
    renames.get(target) match {
      case None      => Seq(this)
      case Some(seq) => seq.map(n => this.copy(target = n))
    }
  }
  override private[firrtl] def dedup: Option[(Any, Annotation, ReferenceTarget)] = this match {
    case a @ DocStringAnnotation(refTarget: ReferenceTarget, _) =>
      Some(((refTarget.pathlessTarget, description), copy(target = refTarget.pathlessTarget), refTarget))
    case a @ DocStringAnnotation(pathTarget: InstanceTarget, _) =>
      Some(((pathTarget.pathlessTarget, description), copy(target = pathTarget.pathlessTarget), pathTarget.asReference))
    case _ => None
  }
}

/**
  * An Verilog-style attribute.
  * @param target the object being given an attribute
  * @param description the attribute
  */
case class AttributeAnnotation(target: Target, description: String) extends DescriptionAnnotation {
  def update(renames: RenameMap): Seq[AttributeAnnotation] = {
    renames.get(target) match {
      case None      => Seq(this)
      case Some(seq) => seq.map(n => this.copy(target = n))
    }
  }
  override private[firrtl] def dedup: Option[(Any, Annotation, ReferenceTarget)] = this match {
    case a @ AttributeAnnotation(refTarget: ReferenceTarget, _) =>
      Some(((refTarget.pathlessTarget, description), copy(target = refTarget.pathlessTarget), refTarget))
    case a @ AttributeAnnotation(pathTarget: InstanceTarget, _) =>
      Some(((pathTarget.pathlessTarget, description), copy(target = pathTarget.pathlessTarget), pathTarget.asReference))
    case _ => None
  }
}

/**
  * Base trait for an object that has associated descriptions
  */
private sealed trait HasDescription {
  def descriptions: Seq[Description]
}

/**
  * Base trait for a description that gives some information about a `FirrtlNode`.
  * Usually, we would like to emit these descriptions in some way.
  */
sealed trait Description extends FirrtlNode

/**
  * A docstring description (a comment)
  * @param string a comment
  */
case class DocString(string: StringLit) extends Description {
  def serialize: String = "@[" + string.serialize + "]"
}

/**
  * A Verilog-style attribute.
  * @param string the attribute
  */
case class Attribute(string: StringLit) extends Description {
  def serialize: String = "@[" + string.serialize + "]"
}

/**
  * A statement with descriptions
  * @param descriptions
  * @param stmt the encapsulated statement
  */
private case class DescribedStmt(descriptions: Seq[Description], stmt: Statement)
    extends Statement
    with HasDescription {
  override def serialize: String = s"${descriptions.map(_.serialize).mkString("\n")}\n${stmt.serialize}"
  def mapStmt(f:       Statement => Statement):   Statement = f(stmt)
  def mapExpr(f:       Expression => Expression): Statement = this.copy(stmt = stmt.mapExpr(f))
  def mapType(f:       Type => Type):             Statement = this.copy(stmt = stmt.mapType(f))
  def mapString(f:     String => String):         Statement = this.copy(stmt = stmt.mapString(f))
  def mapInfo(f:       Info => Info):             Statement = this.copy(stmt = stmt.mapInfo(f))
  def foreachStmt(f:   Statement => Unit):        Unit = f(stmt)
  def foreachExpr(f:   Expression => Unit):       Unit = stmt.foreachExpr(f)
  def foreachType(f:   Type => Unit):             Unit = stmt.foreachType(f)
  def foreachString(f: String => Unit):           Unit = stmt.foreachString(f)
  def foreachInfo(f:   Info => Unit):             Unit = stmt.foreachInfo(f)
}

/**
  * A module with descriptions
  * @param descriptions list of descriptions for the module
  * @param portDescriptions list of descriptions for the module's ports
  * @param mod the encapsulated module
  */
private case class DescribedMod(
  descriptions:     Seq[Description],
  portDescriptions: Map[String, Seq[Description]],
  mod:              DefModule)
    extends DefModule
    with HasDescription {
  val info = mod.info
  val name = mod.name
  val ports = mod.ports
  override def serialize: String = s"${descriptions.map(_.serialize).mkString("\n")}\n${mod.serialize}"
  def mapStmt(f:       Statement => Statement): DefModule = this.copy(mod = mod.mapStmt(f))
  def mapPort(f:       Port => Port):           DefModule = this.copy(mod = mod.mapPort(f))
  def mapString(f:     String => String):       DefModule = this.copy(mod = mod.mapString(f))
  def mapInfo(f:       Info => Info):           DefModule = this.copy(mod = mod.mapInfo(f))
  def foreachStmt(f:   Statement => Unit):      Unit = mod.foreachStmt(f)
  def foreachPort(f:   Port => Unit):           Unit = mod.foreachPort(f)
  def foreachString(f: String => Unit):         Unit = mod.foreachString(f)
  def foreachInfo(f:   Info => Unit):           Unit = mod.foreachInfo(f)
}

/** Wraps modules or statements with their respective described nodes. Descriptions come from [[DescriptionAnnotation]].
  * Describing a module or any of its ports will turn it into a `DescribedMod`. Describing a Statement will turn it into
  * a (private) `DescribedStmt`.
  *
  * @note should only be used by VerilogEmitter, described nodes will
  *       break other transforms.
  */
class AddDescriptionNodes extends Transform with DependencyAPIMigration {

  override def prerequisites = firrtl.stage.Forms.LowFormMinimumOptimized ++
    Seq(
      Dependency[firrtl.transforms.BlackBoxSourceHelper],
      Dependency[firrtl.transforms.FixAddingNegativeLiterals],
      Dependency[firrtl.transforms.ReplaceTruncatingArithmetic],
      Dependency[firrtl.transforms.InlineBitExtractionsTransform],
      Dependency[firrtl.transforms.PropagatePresetAnnotations],
      Dependency[firrtl.transforms.InlineAcrossCastsTransform],
      Dependency[firrtl.transforms.LegalizeClocksAndAsyncResetsTransform],
      Dependency[firrtl.transforms.FlattenRegUpdate],
      Dependency(passes.VerilogModulusCleanup),
      Dependency[firrtl.transforms.VerilogRename],
      Dependency(firrtl.passes.VerilogPrep)
    )

  override def optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform) = false

  def onStmt(compMap: Map[String, Seq[Description]])(stmt: Statement): Statement = {
    val s = stmt.map(onStmt(compMap))
    val sname = s match {
      case d: IsDeclaration => Some(d.name)
      case _ => None
    }
    val descs = sname.flatMap({
      case name =>
        compMap.get(name)
    })
    (descs, s) match {
      case (Some(d), DescribedStmt(prevDescs, ss)) => DescribedStmt(prevDescs ++ d, ss)
      case (Some(d), ss)                           => DescribedStmt(d, ss)
      case (None, _)                               => s
    }
  }

  def onModule(
    modMap:   Map[String, Seq[Description]],
    compMaps: Map[String, Map[String, Seq[Description]]]
  )(mod:      DefModule
  ): DefModule = {
    val compMap = compMaps.getOrElse(mod.name, Map())
    val newMod = mod.mapStmt(onStmt(compMap))
    val portDesc = mod.ports.collect {
      case p @ Port(_, name, _, _) if compMap.contains(name) =>
        name -> compMap(name)
    }.toMap

    val modDesc = modMap.get(newMod.name).getOrElse(Seq())

    if (portDesc.nonEmpty || modDesc.nonEmpty) {
      DescribedMod(modDesc, portDesc, newMod)
    } else {
      newMod
    }
  }

  /**
    * Merges descriptions of like types.
    *
    * Multiple DocStrings on the same object get merged together into one big multi-line comment.
    * Similarly, multiple attributes on the same object get merged into one attribute with attributes separated by
    * commas.
    * @param descs List of `Description`s that are modifying the same object
    * @return List of `Description`s with some descriptions merged
    */
  def mergeDescriptions(descs: Seq[Description]): Seq[Description] = {
    val (docs: Seq[DocString] @unchecked, nodocs) = descs.partition {
      case _: DocString => true
      case _ => false
    }
    val (attrs: Seq[Attribute] @unchecked, rest) = nodocs.partition {
      case _: Attribute => true
      case _ => false
    }

    val doc = if (docs.nonEmpty) {
      Seq(DocString(StringLit.unescape(docs.map(_.string.string).mkString("\n\n"))))
    } else {
      Seq()
    }
    val attr = if (attrs.nonEmpty) {
      Seq(Attribute(StringLit.unescape(attrs.map(_.string.string).mkString(", "))))
    } else {
      Seq()
    }

    rest ++ doc ++ attr
  }

  def collectMaps(
    annos: Seq[Annotation]
  ): (Map[String, Seq[Description]], Map[String, Map[String, Seq[Description]]]) = {
    val modList = annos.collect {
      case DocStringAnnotation(ModuleTarget(_, m), desc) => (m, DocString(StringLit.unescape(desc)))
      case AttributeAnnotation(ModuleTarget(_, m), desc) => (m, Attribute(StringLit.unescape(desc)))
    }

    // map field 1 (module name) -> field 2 (a list of Descriptions)
    val modMap = modList
      .groupBy(_._1)
      .mapValues(_.map(_._2))
      // and then merge like descriptions (e.g. multiple docstrings into one big docstring)
      .mapValues(mergeDescriptions)

    val compList = annos.collect {
      case DocStringAnnotation(ReferenceTarget(_, m, _, c, _), desc) =>
        (m, c, DocString(StringLit.unescape(desc)))
      case AttributeAnnotation(ReferenceTarget(_, m, _, c, _), desc) =>
        (m, c, Attribute(StringLit.unescape(desc)))
    }

    // map field 1 (name) -> a map that we build
    val compMap = compList
      .groupBy(_._1)
      .mapValues(
        // map field 2 (component name) -> field 3 (a list of Descriptions)
        _.groupBy(_._2)
          .mapValues(_.map(_._3))
          // and then merge like descriptions (e.g. multiple docstrings into one big docstring)
          .mapValues(mergeDescriptions)
          .toMap
      )

    (modMap.toMap, compMap.toMap)
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
