package chisel3


import scala.annotation.tailrec
import scala.collection.mutable

/** A prefix tree. Prefixes represented by a list of `Token` values. For each
  * valid path the tree will contain a value of type `Value`.
  *
  * To use this, do not extend this trait (it is sealed). Instead, create an empty trie with
  * the helper method in the object. Then, add items with the insert method.
  */
sealed trait Trie[Token, Value] {
  def value: Option[Value]
  protected def setValue(value: Value): Unit
  protected def clearValue(): Unit
  protected def children:     mutable.LinkedHashMap[Token, Trie[Token, Value]]

  /** Inserts a value into the trie */
  @tailrec
  final def insert(tokens: Seq[Token], value: Value): Unit = {
    if (tokens.isEmpty) {
      setValue(value)
    } else {
      val child = children.getOrElseUpdate(tokens.head, Trie.empty)
      child.insert(tokens.tail, value)
    }
  }

  /** Deletes a value in the trie */
  final def delete(tokens: Seq[Token]): Unit = {
    if (tokens.isEmpty) {
      clearValue()
    } else if (children.get(tokens.head).nonEmpty) {
      val child = children(tokens.head)
      child.delete(tokens.tail)
      if (child.children.isEmpty && child.value.isEmpty) children.remove(tokens.head)
    }
  }

  /** Returns the child located at the path */
  @tailrec
  final def getChild(path: Seq[Token]): Option[Trie[Token, Value]] = {
    if (path.isEmpty) {
      Some(this)
    } else {
      if (children.contains(path.head)) {
        children(path.head).getChild(path.tail)
      } else {
        None
      }
    }
  }

  /** Returns the first value found, along the path */
  @tailrec
  final def getFirst(path: Seq[Token]): Option[Value] = {
    value match {
      case Some(v)                              => Some(v)
      case None if path.isEmpty                 => None
      case None if children.contains(path.head) => children(path.head).getFirst(path.tail)
      case None                                 => None
    }
  }

  /** Returns a new Trie, with each value being the result of applying f
    * @param f Given the path and optional value, return a new optional value for the new trie
    * @return
    */
  final def transform[NewValue](f: (Seq[Token], Option[Value]) => Option[NewValue]): Trie[Token, NewValue] = {
    val newTrie = Trie.empty[Token, NewValue]

    def recTransform(tokens: Seq[Token], subtrie: Trie[Token, Value]): Unit = {
      val newValue = f(tokens, subtrie.value)
      newValue.map(x => newTrie.insert(tokens, x))
      subtrie.children.keys.foreach { case token => recTransform(tokens :+ token.asInstanceOf[Token], subtrie.children(token)) }
    }

    recTransform(Nil, this)
    newTrie
  }

  /** Returns local children to this Trie */
  final def getChildren(): List[(Token, Trie[Token, Value])] = children.toList

  /** Returns children of the Trie located at this path */
  final def getChildren(path: Seq[Token]): List[(Token, Trie[Token, Value])] = {
    if (path.isEmpty) getChildren()
    else {
      children(path.head).getChildren(path.tail)
    }
  }

  /** Returns the optional value at the path */
  @tailrec
  final def get(path: Seq[Token]): Option[Value] = {
    if (path.isEmpty) value
    else {
      children.get(path.head) match {
        case None       => None
        case Some(trie) => trie.get(path.tail)
      }
    }
  }

  private final def collectDeepWithPath[T](path: Seq[Token])(collector: PartialFunction[(Seq[Token], Option[Value]), T]): Iterable[T] = {
    val myItems           = collector.lift((path, value))
    val deepChildrenItems = getChildren().flatMap { case (token, c) =>
      c.collectDeepWithPath(path :+ token)(collector)
    }
    myItems ++ deepChildrenItems
  }

  /** For each matching path/value provided in the partial function, return an item */
  final def collectDeep[T](collector: PartialFunction[(Seq[Token], Option[Value]), T]): Iterable[T] = {
    collectDeepWithPath(Nil)(collector)
  }

}

object Trie {

  /** creates a new empty Trie
    */
  def empty[Token, Value]: Trie[Token, Value] = {
    new Trie[Token, Value] {
      var value: Option[Value] = None
      def setValue(valuex: Value): Unit = {
        value = Some(valuex)
      }
      def clearValue(): Unit   = value = None
      val children             = mutable.LinkedHashMap.empty[Token, Trie[Token, Value]]
    }
  }
}