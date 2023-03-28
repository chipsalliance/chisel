package chisel3.internal

private[internal] trait CloneToContext {
  def cloneTo(c: Context): CloneToContext
}

// Datastructure to represent contexts (with back pointers to "provenance", or version of this context with less context information)
private[chisel3] class Context(val key: String, val parent: Option[Context], val provenance: Option[Context]) {
  private val localChildrenKeys = new collection.mutable.LinkedHashSet[String]()
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
  // entire parent context path
  lazy val parentContextPath: Seq[Context] = (this +: parent.toSeq.flatMap(_.parentContextPath))
  // entire provenance context path
  lazy val provenanceContextPath: Seq[Context] = (this +: provenance.toSeq.flatMap(_.provenanceContextPath))

  /* Helpers to determine whether this Context has these properties */

  lazy val isTop = parent.isEmpty
  lazy val isOrigin = origin == this
  lazy val isTemplate = template == this

  /* Adding children to origin new Contexts */

  private def newChildKey(k: String): Unit = {
    if (isOrigin) localChildrenKeys += k else provenance.map(_.newChildKey(k))
  }

  def instantiateChild(childKey: String, provenance: Context): Context = {
    require(isOrigin)
    require(children.get(childKey).isEmpty)
    newChildKey(childKey)
    val child = new Context(childKey, Some(this), Some(provenance))
    children.put(childKey, child)
    child
  }

  /* Accessing contexts */

  def apply(childKey: String): Context = getChild(childKey).get

  def getChild(childKey: String): Option[Context] = getOrDeriveChild(childKey)

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

  private def recurseParent(p: Context, top: Context): Option[Context] = {
    if (p.origin == top) Some(p)
    else {
      if (p == p.top) None
      else {
        recurseParent(p.parent.get, top).map(_.apply(p.key))
      }
    }
  }

  // either we share a top, or we don't
  // If we share a top, return
  // If we don't, then either
  //  - (1) this.top subsumes other.top
  //    - other.top is my provenance
  //    - If any origin parent path == other.top, then retop
  //  - (2) this.top is subsumed by other.top
  //    - None
  //  - (3) this.top and other.top have no relationship
  //    - NoTone
  def copyTo(other: Context): Context = {
    if (other.top == top) {
      this
    } else {
      if (other.origin == top) {
        val newMe = parentPath.reverse.tail.foldLeft(other) { (p, key) =>
          p.apply(key)
        }
        newMe
      } else {
        recurseParent(other, top).getOrElse(this)
      }
    }
  }

  /* Values */

  private var valueVar: Option[CloneToContext] = None
  def setValue[T <: CloneToContext](v: T): T = {
    require(valueVar.isEmpty)
    valueVar = Some(v)
    v
  }

  def getOrSetValue[T <: CloneToContext](v: => T): T = if (hasValue) value.asInstanceOf[T] else setValue(v)

  def value: CloneToContext = {
    require(hasValue, s"No value: cannot call value on $this:\n$visualize")
    getValue.get
  }
  def getValue: Option[CloneToContext] = valueVar.orElse(provenance.flatMap(_.getValue).map(x => x.cloneTo(this)))
  def hasLocalValue = valueVar.isDefined
  def hasValue = getValue.isDefined

  // Helpful for debugging
  def id = this.hashCode().toString.take(3)
  def keyWithDefinition = if (isTemplate) key else key + ":" + template.key
  def keyWithId = keyWithDefinition + "(" + id + ")"

  /* Helpers to visualize the Context tree information */

  def parentPath: Seq[String] = (key +: parent.toSeq.flatMap(_.parentPath))

  def provenancePath: Seq[String] = (key +: provenance.toSeq.flatMap(_.provenancePath))

  def parentageAndProvenance(getKey: Context => String): String = {
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

  def visualize = parentageAndProvenance(_.key)
  def visualizeWithId = parentageAndProvenance(_.keyWithId)
  def visualizeWithDefinition = parentageAndProvenance(_.keyWithDefinition)

  def target:               String = parentContextPath.map(_.key).reverse.mkString("/")
  def targetWithId:         String = parentContextPath.map(_.keyWithId).reverse.mkString("/")
  def targetWithDefinition: String = parentContextPath.map(_.keyWithId).reverse.mkString("/")

  //Static
  def align(txt: String, size: Int): String = {
    require(size >= txt.size, s"$size is not large enough to contain ${txt.size}")
    txt + (" " * (size - txt.size))
  }

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
