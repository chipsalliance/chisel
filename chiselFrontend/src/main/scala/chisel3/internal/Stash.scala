// See LICENSE for license details.

package chisel3.internal

import chisel3.experimental.BaseModule

import scala.collection.mutable

case class Link(originalModule: Long, wrapperModule: Long, parentModule: Long)

//private[chisel3] object Stash {
object Stash {

  // Stash of either elaborated to currently elaborating modules
  // Additionally, has accessors/setters
  private val moduleStash = mutable.HashMap[Long, BaseModule]()
  def updateStash(id: Long, module: BaseModule): Unit = {
    require(!moduleStash.contains(id), s"Cannot add module ${module.name} to stash because its id $id conflicts with ${moduleStash(id).name}.")
    moduleStash(id) = module
  }
  def clearStash(): Unit = moduleStash.clear()
  def initializeStash(stash: Map[Long, BaseModule]): Unit = {
    require(moduleStash.isEmpty, s"Cannot initialize a non-empty stash!")
    moduleStash ++= stash
  }
  def get(id: Long): Option[BaseModule] = moduleStash.get(id)
  def module(id: Long): BaseModule = {
    require(moduleStash.contains(id), s"Module with id $id is not contained in stash!")
    moduleStash(id)
  }

  // Maps current child/parents relationships in design
  private val parentMap = mutable.HashMap[Long, Seq[Long]]()
  def getParents(moduleId: Long): Seq[Long] = {
    parentMap.getOrElse(moduleId, Nil)
  }
  def setParent(child: Long, parent: Long) = {
    val parents = getParents(child)
    assert(child != parent)
    parentMap(child) = parent +: parents
  }

  // Tracks which parent is the "active" parent, if a child has been
  //   imported multiple times
  private val activeParentMap = mutable.HashMap[Long, Long]()
  def setActiveParent(id: Long, parent: Long): Unit = {
    assert(id != parent)
    require(!activeParentMap.contains(id) || activeParentMap(id) == parent, s"Cannot set active parent $parent for $id, as it already has parent ${activeParentMap(id)}")
    activeParentMap(id) = parent
  }
  def clearActiveParent(id: Long): Unit = {
    activeParentMap.remove(id)
  }
  def getActiveParent(id: Long): Option[Long] = {
    activeParentMap.get(id)
  }

  // Maps modules with their imported versions
  private val linkedModuleMap = mutable.HashMap[Long, Set[Long]]()
  def setLink(original: Long, imported: Long): Unit = {
    linkedModuleMap(original) = linkedModuleMap.getOrElse(original, Set.empty) + imported
  }
  def clearLink(original: Long, imported: Long): Unit = {
    linkedModuleMap(original) = linkedModuleMap.getOrElse(original, Set.empty) - imported
  }
  def getLinks(original: Long): Set[Long] = {
    linkedModuleMap.getOrElse(original, Set.empty)
  }

  // Tracks which linked module is the "active" linked module, if a module has been
  //   imported multiple times
  private val activeLinkMap = mutable.HashMap[Long, Long]()
  def setActiveLink(original: Long, imported: Long): Unit = {
    assert(original != imported)
    require(!activeLinkMap.contains(original) || activeLinkMap(original) == imported, s"Cannot set active link $imported for $original, as it already has imported ${activeLinkMap(original)}")
    activeLinkMap(original) = imported
    //println("Set", activeLinkMap)
  }
  def clearActiveLink(original: Long): Unit = {
    //println("Remove", activeLinkMap)
    activeLinkMap.remove(original)
  }
  def getActiveLink(original: Long): Option[Long] = {
    //println("Get", activeLinkMap)
    activeLinkMap.get(original)
  }

  def printStatus() = {
    val stackElts = Thread.currentThread().getStackTrace()
      .reverse  // so stack frame numbers are deterministic across calls
      .dropRight(2)  // discard Thread.getStackTrace and updateBundleStack
    println("CurrentModule:", Builder.currentModule, Builder.currentModule.map(_._id))
    println("Module Stash:")
    moduleStash.foreach(x => println(s"    $x"))
    println("Active Parents:")
    activeParentMap.foreach(x => println(s"    $x"))
    println("Active Links:")
    activeLinkMap.foreach(x => println(s"    $x"))
    println("Parents:")
    parentMap.foreach(x => println(s"    $x"))
    println("Links:")
    linkedModuleMap.foreach(x => println(s"    $x"))
    println("StackTrace:")
    stackElts.foreach(x => println(s"    $x"))
  }
}

