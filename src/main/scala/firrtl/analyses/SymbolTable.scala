// SPDX-License-Identifier: Apache-2.0

package firrtl.analyses

import firrtl.ir._
import firrtl.passes.MemPortUtils
import firrtl.{InstanceKind, Kind, WDefInstance}

import scala.collection.mutable

/** This trait represents a data structure that stores information
  * on all the symbols available in a single firrtl module.
  * The module can either be scanned all at once using the
  * scanModule helper function from the companion object or
  * the SymbolTable can be updated while traversing the module by
  * calling the declare method every time a declaration is encountered.
  * Different implementations of SymbolTable might want to store different
  * information (e.g., only the names without the types) or build
  * different indices depending on what information the transform needs.
  */
trait SymbolTable {
  // methods that need to be implemented by any Symbol table
  def declare(name:         String, tpe:    Type, kind: Kind): Unit
  def declareInstance(name: String, module: String): Unit

  // convenience methods
  def declare(d: DefInstance): Unit = declareInstance(d.name, d.module)
  def declare(d: DefMemory):   Unit = declare(d.name, MemPortUtils.memType(d), firrtl.MemKind)
  def declare(d: DefNode):     Unit = declare(d.name, d.value.tpe, firrtl.NodeKind)
  def declare(d: DefWire):     Unit = declare(d.name, d.tpe, firrtl.WireKind)
  def declare(d: DefRegister): Unit = declare(d.name, d.tpe, firrtl.RegKind)
  def declare(d: Port):        Unit = declare(d.name, d.tpe, firrtl.PortKind)
}

/** Trusts the type annotation on DefInstance nodes instead of re-deriving the type from
  * the module ports which would require global (cross-module) information.
  */
private[firrtl] abstract class LocalSymbolTable extends SymbolTable {
  def declareInstance(name: String, module: String): Unit = declare(name, UnknownType, InstanceKind)
  override def declare(d:   WDefInstance): Unit = declare(d.name, d.tpe, InstanceKind)
}

/** Uses a function to derive instance types from module names */
private[firrtl] abstract class ModuleTypesSymbolTable(moduleTypes: String => Type) extends SymbolTable {
  def declareInstance(name: String, module: String): Unit = declare(name, moduleTypes(module), InstanceKind)
}

/** Uses a single buffer. No O(1) access, but deterministic Symbol order. */
private[firrtl] trait WithSeq extends SymbolTable {
  private val symbols = mutable.ArrayBuffer[Symbol]()
  override def declare(name: String, tpe: Type, kind: Kind): Unit = symbols.append(Sym(name, tpe, kind))
  def getSymbols: Iterable[Symbol] = symbols
}

/** Uses a mutable map to provide O(1) access to symbols by name. */
private[firrtl] trait WithMap extends SymbolTable {
  private val symbols = mutable.HashMap[String, Symbol]()
  override def declare(name: String, tpe: Type, kind: Kind): Unit = {
    assert(!symbols.contains(name), s"Symbol $name already declared: ${symbols(name)}")
    symbols(name) = Sym(name, tpe, kind)
  }
  def apply(name: String): Symbol = symbols(name)
  def size: Int = symbols.size
}

private case class Sym(name: String, tpe: Type, kind: Kind) extends Symbol
private[firrtl] trait Symbol { def name: String; def tpe: Type; def kind: Kind }

/** only remembers the names of symbols */
private[firrtl] class NamespaceTable extends LocalSymbolTable {
  private var names = List[String]()
  override def declare(name: String, tpe: Type, kind: Kind): Unit = names = name :: names
  def getNames: Seq[String] = names
}

/** Provides convenience methods to populate SymbolTables. */
object SymbolTable {
  def scanModule[T <: SymbolTable](m: DefModule, t: T): T = {
    implicit val table: T = t
    m.foreachPort(table.declare)
    m.foreachStmt(scanStatement)
    table
  }
  private def scanStatement(s: Statement)(implicit table: SymbolTable): Unit = s match {
    case d: DefInstance => table.declare(d)
    case d: DefMemory   => table.declare(d)
    case d: DefNode     => table.declare(d)
    case d: DefWire     => table.declare(d)
    case d: DefRegister => table.declare(d)
    // Matches named statements like printf, stop, assert, assume, cover if the name is not empty.
    // Empty names are allowed for backwards compatibility reasons and
    // indicate that the entity has essentially no name.
    case s: IsDeclaration if s.name.nonEmpty => table.declare(s.name, UnknownType, firrtl.UnknownKind)
    case other => other.foreachStmt(scanStatement)
  }
}
