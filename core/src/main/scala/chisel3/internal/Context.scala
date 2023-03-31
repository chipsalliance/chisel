// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

/** All values of contexts must cloneable to a top with more context */
private[chisel3] trait CloneToContext {
  def cloneTo(c: Context): CloneToContext = this // For now this does not work, but it is only necessary when box-types are a thing
}

/** Data-structure to represent hierarchical contexts
  *
  * Contains links from parent-to-child and child-to-parent
  * Contains links to its provenance, i.e. from instance to its definition or its version of itself referenced with less context
  *
  * E.g. For context Top/foo:Foo/bar:Bar (with child baz:Baz), it contains pointers to the following contexts:
  *  - Parent:     Top/foo:Foo
  *  - Children:   Top/foo:Foo/bar:Bar/baz:Baz
  *  - Provenance: Foo/bar:Bar
  *
  * @param key identifier used to pick which context's children to select
  * @param parent if a parent is present; if None, this is a "top" context
  * @param provenance if a provenance is preset; if None, this is an "origin" context
  */
private[chisel3] class Context private (val key: String, val parent: Option[Context], val provenance: Option[Context]) {
  // Cache of children keys that this context has
  private val localChildrenKeys = new collection.mutable.LinkedHashSet[String]()
  // Cache of children Contexts
  private val children = new collection.mutable.LinkedHashMap[String, Context]()

  /** At this moment in time, all known children keys of this or any provenance Contexts */
  def childrenKeys: List[String] = if (isOrigin) localChildrenKeys.toList else provenance.map(_.childrenKeys).get
  def isChild(childKey: String): Boolean = childrenKeys.contains(childKey)

  /* Accessors to important contexts, relative to this */

  // Top-most context
  lazy val top: Context = parent.map(_.top).getOrElse(this)
  // Provenance origin, could have a different top context
  lazy val origin: Context = provenance.map(_.origin).getOrElse(this)
  // Provenance origin, but who shares same top context
  lazy val template: Context = {
    if (Some(top) == provenance.map(_.top)) provenance.get.template else this
  }
  // Entire parent context path
  lazy val parentContextPath: Seq[Context] = (this +: parent.toSeq.flatMap(_.parentContextPath))
  // Entire provenance context path
  lazy val provenanceContextPath: Seq[Context] = (this +: provenance.toSeq.flatMap(_.provenanceContextPath))

  /* Helpers to determine whether this Context has these properties */

  lazy val isTop = parent.isEmpty
  lazy val isOrigin = origin == this
  lazy val isTemplate = template == this

  /* Creates a new child key to `this.origin` */
  private def createNewOriginChildKey(k: String): Unit = {
    if (isOrigin) localChildrenKeys += k else provenance.map(_.createNewOriginChildKey(k))
  }

  /* Creates a new child Context of `this`; can only call when `this.isOrigin` is true */
  def instantiateChild(childKey: String, provenance: Context): Context = {
    require(isOrigin)
    require(children.get(childKey).isEmpty)
    createNewOriginChildKey(childKey)
    val child = new Context(childKey, Some(this), Some(provenance))
    children.put(childKey, child)
    child
  }
  // Necessary if instance == definition (pre D/I world)
  def instantiateOriginChild(childKey: String): Context = {
    require(isOrigin)
    require(children.get(childKey).isEmpty)
    createNewOriginChildKey(childKey)
    val child = new Context(childKey, Some(this), None)
    children.put(childKey, child)
    child
  }
  def instantiateOriginChildWithValue(childKey: String, value: CloneToContext): Context = {
    val c = instantiateOriginChild(childKey)
    c.setValue(value)
    c
  }

  /** Return the appropriate child of `this`, error if this child key is not defined `this.origin` */
  def apply(childKey: String): Context = getChild(childKey).get

  /** Optionally return the appropriate child of `this` if defined `this.origin` */
  def getChild(childKey: String): Option[Context] = getOrDeriveChild(childKey)

  /** Accessing children contexts from `this`; either hit in the children cache, or build, cache, and return a new Context with provenance back to its origin */
  private def getOrDeriveChild(childKey: String): Option[Context] = {
    if (childrenKeys.contains(childKey)) {
      if (children.contains(childKey)) {
        val c = children(childKey)
        Some(c)
      } else {
        val computedProvenance = provenance.flatMap(s => s.getOrDeriveChild(childKey))
        val ret = new Context(childKey, Some(this), computedProvenance)
        children.put(childKey, ret)
        Some(ret)
      }
    } else None
  }

  /** Computes a new context to take the place of the previous `top`, rooted at the same top as `p` */
  private def computeNewTop(p: Context, top: Context): Option[Context] = {
    if (top == p.origin) {
      Some(p)
    } else {
      if (p == p.top) None else computeNewTop(p.parent.get, top)
    }
  }

  /** Retops `this` to the top of `other` if `this.top` is subsumed by `origin` or any of its parents
    */
  def copyTo(other: Context): Context = {
    if (top == other.top) { // Already share same top-level context, so nothing needed
      this
    } else { // We need to retop this to a parent of other, if we can find one. Otherwise, return this unchanged
      computeNewTop(other, top).map { newTop =>
        parentPath.reverse.tail.foldLeft(newTop) { (p, key) =>
          p.apply(key)
        }
      }.getOrElse(this)
    }
  }

  /** ***** Context-specific values *******
    */

  private var valueVar: Option[CloneToContext] = None
  def setValue[T <: CloneToContext](v: T): T = {
    require(valueVar.isEmpty)
    valueVar = Some(v)
    v
  }

  /** Returns the first value defined along the provenance path; if its not defined sets the local value and returns it */
  def getOrSetValue[T <: CloneToContext](v: => T): T = if (hasValue) value.asInstanceOf[T] else setValue(v)

  /** Returns the first value defined along the provenance path; if its not defined sets the local value and returns it */
  def getLocalOrSetValue[T <: CloneToContext](v: => T): T = if (hasLocalValue) value.asInstanceOf[T] else setValue(v)

  /** Returns the first value defined along the provenance path; errors if no value is present along provenance path */
  def value: CloneToContext = {
    require(hasValue, s"No value: cannot call value on $this:\n$visualize")
    getValue.get
  }

  /** Optionally returns the first value defined along the provenance path; e.g. values with more top-context have higher priority */
  def getValue: Option[CloneToContext] = valueVar.orElse(provenance.flatMap(_.getValue).map(x => x.cloneTo(this)))

  /** If any value is defined in provenance chain */
  def hasValue = getValue.isDefined

  /** Optionally returns the value defined on `this`; does not look down provenance chain */
  def getLocalValue: Option[CloneToContext] = valueVar

  /** If defined on `this`; does not look down provenance chain */
  def hasLocalValue = valueVar.isDefined

  /** ***** Debugging and Visualization Functions *******
    */

  /** Indicates this specific context object in memory; helpful for debugging */
  def id = this.hashCode().toString.take(3)

  /** Associates `this.key` with its `template.key` */
  def keyWithDefinition = if (isTemplate) key else key + ":" + template.key

  /** Associates `this.key` with its `template.key` and `this.id` */
  def keyWithId = keyWithDefinition + "(" + id + ")"

  /** List of parent key's in order from `this` to `top` */
  def parentPath: Seq[String] = (key +: parent.toSeq.flatMap(_.parentPath))

  /** List of provenance key's in order from `this` to `origin` */
  def provenancePath: Seq[String] = (key +: provenance.toSeq.flatMap(_.provenancePath))

  /** A matrix of parent (vertical) and provenance (horizontal) chains customizable by how to translate each Context to a String */
  def visualizeParentsAndProvenance(getKey: Context => String): String = {
    def align(txt: String, size: Int): String = {
      require(size >= txt.size, s"$size is not large enough to contain ${txt.size}")
      txt + (" " * (size - txt.size))
    }
    val sizes = collection.mutable.HashMap.empty[Int, Int]
    val parentProvenance: Seq[Seq[String]] =
      parentContextPath.map { p =>
        p.provenanceContextPath.zipWithIndex.map {
          case (x, i) =>
            val k = getKey(x)
            sizes(i) = k.size.max(sizes.getOrElse(i, 0))
            k
        }
      }
    parentProvenance.reverse.map { ls =>
      ls.zipWithIndex.map { case (x, i) => align(x, sizes(i)) }.mkString(" # ")
    }.mkString("\n")
  }

  /** A matrix of parent (vertical) and provenance (horizontal) keys */
  def visualize = visualizeParentsAndProvenance(_.key)

  /** A matrix of parent (vertical) and provenance (horizontal) keys with id */
  def visualizeWithId = visualizeParentsAndProvenance(_.keyWithId)

  /** A matrix of parent (vertical) and provenance (horizontal) keys with origin.key and id */
  def visualizeWithDefinition = visualizeParentsAndProvenance(_.keyWithDefinition)

  /** Parent chain of keys */
  def target: String = parentContextPath.map(_.key).reverse.mkString("/")

  /** Parent chain of keys with id */
  def targetWithId: String = parentContextPath.map(_.keyWithId).reverse.mkString("/")

  /** Parent chain of keys with id and origin.key */
  def targetWithDefinition: String = parentContextPath.map(_.keyWithId).reverse.mkString("/")

  /* Iterators to walk Context trees */

  def iterateDown[T](depth: Int, f: PartialFunction[(Int, Context), Unit], descend: (Context => Boolean)): Unit = {
    f.lift((depth, this))
    if (descend(this)) {
      childrenKeys.foreach {
        case k =>
          val c = getChild(k).get
          c.iterateDown(depth + 1, f, descend)
      }
    }
  }

  def iterateUp[T](depth: Int, f: PartialFunction[(Int, Context), Unit], ascend: (Context => Boolean)): Unit = {
    f.lift((depth, this))
    if (ascend(this)) {
      parent.foreach {
        case c =>
          c.iterateUp(depth + 1, f, ascend)
      }
    }
  }
}

object Context {
  def apply(key: String): Context = new Context(key, None, None)
}
