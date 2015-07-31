package Chisel

abstract class Namespace {
  final def name(n: String): String = {
    if (nameExists(n)) name(rename(n))
    else {
      names += n
      n
    }
  }

  def nameExists(n: String): Boolean

  protected final def nameExistsHere(n: String): Boolean =
    names contains n

  private def rename(n: String): String = {
    i += 1
    s"${n}_${i}"
  }

  private var i = 0L
  protected val names = collection.mutable.HashSet[String]()
}

class RootNamespace(initialNames: String*) extends Namespace {
  names ++= initialNames
  def nameExists(n: String) = nameExistsHere(n)
}

class ChildNamespace(parent: Namespace) extends Namespace {
  def nameExists(n: String) = nameExistsHere(n) || parent.nameExists(n)
}

class FIRRTLNamespace extends RootNamespace("mem", "node", "wire", "reg", "inst")
