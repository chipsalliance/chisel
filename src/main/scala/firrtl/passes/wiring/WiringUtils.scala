// See LICENSE for license details.

package firrtl.passes
package wiring

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import scala.collection.mutable
import firrtl.annotations._
import WiringUtils._

/** Declaration kind in lineage (e.g. input port, output port, wire)
  */
sealed trait DecKind
case object DecInput extends DecKind
case object DecOutput extends DecKind
case object DecWire extends DecKind

/** A lineage tree representing the instance hierarchy in a design
  */
case class Lineage(
    name: String,
    children: Seq[(String, Lineage)] = Seq.empty,
    source: Boolean = false,
    sink: Boolean = false,
    sourceParent: Boolean = false,
    sinkParent: Boolean = false,
    sharedParent: Boolean = false,
    addPort: Option[(String, DecKind)] = None,
    cons: Seq[(String, String)] = Seq.empty) {

  def map(f: Lineage => Lineage): Lineage =
    this.copy(children = children.map{ case (i, m) => (i, f(m)) })

  override def toString: String = shortSerialize("")

  def shortSerialize(tab: String): String = s"""
    |$tab name: $name,
    |$tab children: ${children.map(c => tab + "   " + c._2.shortSerialize(tab + "    "))}
    |""".stripMargin

  def foldLeft[B](z: B)(op: (B, (String, Lineage)) => B): B = 
    this.children.foldLeft(z)(op)

  def serialize(tab: String): String = s"""
    |$tab name: $name,
    |$tab source: $source,
    |$tab sink: $sink,
    |$tab sourceParent: $sourceParent,
    |$tab sinkParent: $sinkParent,
    |$tab sharedParent: $sharedParent,
    |$tab addPort: $addPort
    |$tab cons: $cons
    |$tab children: ${children.map(c => tab + "   " + c._2.serialize(tab + "    "))}
    |""".stripMargin
}




object WiringUtils {
  type ChildrenMap = mutable.HashMap[String, Seq[(String, String)]]

  /** Given a circuit, returns a map from module name to children
    * instance/module names
    */
  def getChildrenMap(c: Circuit): ChildrenMap = {
    val childrenMap = new ChildrenMap()
    def getChildren(mname: String)(s: Statement): Statement = s match {
      case s: WDefInstance => 
        childrenMap(mname) = childrenMap(mname) :+ (s.name, s.module)
        s
      case s: DefInstance => 
        childrenMap(mname) = childrenMap(mname) :+ (s.name, s.module)
        s
      case s => s map getChildren(mname)
    }
    c.modules.foreach{ m =>
      childrenMap(m.name) = Nil
      m map getChildren(m.name)
    }
    childrenMap
  }

  /** Counts the number of instances of a module declared under a top module
    */
  def countInstances(childrenMap: ChildrenMap, top: String, module: String): Int = {
    if(top == module) 1
    else childrenMap(top).foldLeft(0) { case (count, (i, child)) =>
      count + countInstances(childrenMap, child, module)
    }
  }

  /** Returns a module's lineage, containing all children lineages as well
    */
  def getLineage(childrenMap: ChildrenMap, module: String): Lineage =
    Lineage(module, childrenMap(module) map { case (i, m) => (i, getLineage(childrenMap, m)) } )

  /** Sets the sink, sinkParent, source, and sourceParent fields of every
    *  Lineage in tree
    */
  def setFields(sinks: Set[String], source: String)(lin: Lineage): Lineage = lin map setFields(sinks, source) match {
    case l if sinks.contains(l.name) => l.copy(sink = true)
    case l => 
      val src = l.name == source
      val sinkParent = l.children.foldLeft(false) { case (b, (i, m)) => b || m.sink || m.sinkParent }
      val sourceParent = if(src) true else l.children.foldLeft(false) { case (b, (i, m)) => b || m.source || m.sourceParent }
      l.copy(sinkParent=sinkParent, sourceParent=sourceParent, source=src)
  }

  /** Sets the sharedParent of lineage top
    */
  def setSharedParent(top: String)(lin: Lineage): Lineage = lin map setSharedParent(top) match {
    case l if l.name == top => l.copy(sharedParent = true)
    case l => l
  }

  /** Sets the addPort and cons fields of the lineage tree
    */
  def setThings(portNames:Map[String, String], compName: String)(lin: Lineage): Lineage = {
    val funs = Seq(
      ((l: Lineage) => l map setThings(portNames, compName)),
      ((l: Lineage) => l match {
        case Lineage(name, _, _, _, _, _, true, _, _) => //SharedParent
          l.copy(addPort=Some((portNames(name), DecWire)))
        case Lineage(name, _, _, _, true, _, _, _, _) => //SourceParent
          l.copy(addPort=Some((portNames(name), DecOutput)))
        case Lineage(name, _, _, _, _, true, _, _, _) => //SinkParent
          l.copy(addPort=Some((portNames(name), DecInput)))
        case Lineage(name, _, _, true, _, _, _, _, _) => //Sink
          l.copy(addPort=Some((portNames(name), DecInput)))
        case l => l
      }),
      ((l: Lineage) => l match {
        case Lineage(name, _, true, _, _, _, _, _, _) => //Source
          val tos = Seq(s"${portNames(name)}")
          val from = compName
          l.copy(cons = l.cons ++ tos.map(t => (t, from)))
        case Lineage(name, _, _, _, true, _, _, _, _) => //SourceParent
          val tos = Seq(s"${portNames(name)}")
          val from = l.children.filter { case (i, c) => c.sourceParent }.map { case (i, c) => s"$i.${portNames(c.name)}" }.head
          l.copy(cons = l.cons ++ tos.map(t => (t, from)))
        case l => l
      }),
      ((l: Lineage) => l match {
        case Lineage(name, _, _, _, _, true, _, _, _) => //SinkParent
          val tos = l.children.filter { case (i, c) => (c.sinkParent || c.sink) && !c.sourceParent } map { case (i, c) => s"$i.${portNames(c.name)}" }
          val from = s"${portNames(name)}"
          l.copy(cons = l.cons ++ tos.map(t => (t, from)))
        case l => l
      })
    )
    funs.foldLeft(lin)((l, fun) => fun(l))
  }

  /** Return a map from module to its lineage in the tree
    */
  def pointToLineage(lin: Lineage): Map[String, Lineage] = {
    val map = mutable.HashMap[String, Lineage]()
    def onLineage(l: Lineage): Lineage = {
      map(l.name) = l
      l map onLineage
    }
    onLineage(lin)
    map.toMap
  }
}
