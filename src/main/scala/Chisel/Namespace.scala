package Chisel

private class Namespace(parent: Option[Namespace], keywords: Option[Set[String]]) {
  private var i = 0L
  private val names = collection.mutable.HashSet[String]()
  def forbidden =  keywords.getOrElse(Set()) ++ names

  private def rename(n: String) = { i += 1; s"${n}_${i}" }

  def contains(elem: String): Boolean = {
    forbidden.contains(elem) ||
      parent.map(_ contains elem).getOrElse(false)
  }

  def name(elem: String): String = {
    val res = if(forbidden contains elem) rename(elem) else elem
    names += res
    res
  }

  def child(ks: Option[Set[String]]): Namespace = new Namespace(Some(this), ks)
  def child: Namespace = new Namespace(Some(this), None)
}

private class FIRRTLNamespace extends Namespace(None, Some(Set("mem", "node", "wire", "reg", "inst")))
