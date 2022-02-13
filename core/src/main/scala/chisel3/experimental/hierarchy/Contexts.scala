package chisel3.experimental.hierarchy

case class Contexts(seq: Seq[PartialFunction[Any, Any]] = Nil) {
  def ++ (c: Contexts): Contexts = Contexts(c.seq ++ seq)
  def append(pf: PartialFunction[Any, Any]): Contexts = Contexts(pf +: seq)
  def apply[V](v: V): V = {
    seq.foldLeft(v) { case (res: V, func: PartialFunction[Any, Any]) => 
        func.lift(res).getOrElse(res).asInstanceOf[V]
    }
  }
}