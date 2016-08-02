package firrtl.passes

import firrtl.ir._
import firrtl.{WRef, WSubAccess, WSubIndex, WSubField}
import firrtl.Mappers._
import firrtl.Utils._
import firrtl.WrappedExpression._
import firrtl.Namespace
import scala.collection.mutable


/** Removes all [[firrtl.WSubAccess]] from circuit
  */
object RemoveAccesses extends Pass {
  def name = "Remove Accesses"

  /** Container for a base expression and its corresponding guard
    */
  case class Location(base: Expression, guard: Expression)

  /** Walks a referencing expression and returns a list of valid references
    * (base) and the corresponding guard which, if true, returns that base.
    * E.g. if called on a[i] where a: UInt[2], we would return:
    *   Seq(Location(a[0], UIntLiteral(0)), Location(a[1], UIntLiteral(1)))
    */
  def getLocations(e: Expression): Seq[Location] = e match {
    case e: WRef => create_exps(e).map(Location(_,one))
    case e: WSubIndex =>
      val ls = getLocations(e.exp)
      val start = get_point(e)
      val end = start + get_size(tpe(e))
      val stride = get_size(tpe(e.exp))
      val lsx = mutable.ArrayBuffer[Location]()
      for (i <- 0 until ls.size) {
        if (((i % stride) >= start) & ((i % stride) < end)) {
          lsx += ls(i)
        }
      }
      lsx
    case e: WSubField =>
      val ls = getLocations(e.exp)
      val start = get_point(e)
      val end = start + get_size(tpe(e))
      val stride = get_size(tpe(e.exp))
      val lsx = mutable.ArrayBuffer[Location]()
      for (i <- 0 until ls.size) {
        if (((i % stride) >= start) & ((i % stride) < end)) { lsx += ls(i) }
      }
      lsx
    case e: WSubAccess =>
      val ls = getLocations(e.exp)
      val stride = get_size(tpe(e))
      val wrap = tpe(e.exp).asInstanceOf[VectorType].size
      val lsx = mutable.ArrayBuffer[Location]()
      for (i <- 0 until ls.size) {
        val c = (i / stride) % wrap
        val basex = ls(i).base
        val guardx = AND(ls(i).guard,EQV(uint(c),e.index))
        lsx += Location(basex,guardx)
      }
      lsx
  }
  /** Returns true if e contains a [[firrtl.WSubAccess]]
    */
  def hasAccess(e: Expression): Boolean = {
    var ret: Boolean = false
    def rec_has_access(e: Expression): Expression = e match {
      case (e:WSubAccess) => { ret = true; e }
      case (e) => e map (rec_has_access)
    }
    rec_has_access(e)
    ret
  }
  def run(c: Circuit): Circuit = {
    def remove_m(m: Module): Module = {
      val namespace = Namespace(m)
      def onStmt(s: Statement): Statement = {
        val stmts = mutable.ArrayBuffer[Statement]()
        def create_temp(e: Expression): Expression = {
          val n = namespace.newTemp
          stmts += DefWire(info(s), n, tpe(e))
          WRef(n, tpe(e), kind(e), gender(e))
        }

        /** Replaces a subaccess in a given male expression
          */
        def removeMale(e: Expression): Expression = e match {
          case (_:WSubAccess| _: WSubField| _: WSubIndex| _: WRef) if (hasAccess(e)) => 
            val rs = getLocations(e)
            val foo = rs.find(x => {x.guard != one})
            foo match {
              case None => error("Shouldn't be here")
              case foo: Some[Location] =>
                val temp = create_temp(e)
                val temps = create_exps(temp)
                def getTemp(i: Int) = temps(i % temps.size)
                for((x, i) <- rs.zipWithIndex) {
                  if (i < temps.size) {
                    stmts += Connect(info(s),getTemp(i),x.base)
                  } else {
                    stmts += Conditionally(info(s),x.guard,Connect(info(s),getTemp(i),x.base),EmptyStmt)
                  }
                }
                temp
            }
          case _ => e
        }

        /** Replaces a subaccess in a given female expression
          */
        def removeFemale(info: Info, loc: Expression): Expression = loc match {
          case (_: WSubAccess| _: WSubField| _: WSubIndex| _: WRef) if (hasAccess(loc)) => 
            val ls = getLocations(loc)
            if (ls.size == 1 & weq(ls(0).guard,one)) loc
            else {
              val temp = create_temp(loc)
              for (x <- ls) { stmts += Conditionally(info,x.guard,Connect(info,x.base,temp),EmptyStmt) }
              temp
            }
          case _ => loc
        }

        /** Recursively walks a male expression and fixes all subaccesses
          * If we see a sub-access, replace it.
          * Otherwise, map to children.
          */
        def fixMale(e: Expression): Expression = e match {
          case w: WSubAccess => removeMale(WSubAccess(w.exp, fixMale(w.index), w.tpe, w.gender))
          //case w: WSubIndex => removeMale(w)
          //case w: WSubField => removeMale(w)
          case x => x map fixMale
        }

        /** Recursively walks a female expression and fixes all subaccesses
          * If we see a sub-access, its index is a male expression, and we must replace it.
          * Otherwise, map to children.
          */
        def fixFemale(e: Expression): Expression = e match {
          case w: WSubAccess => WSubAccess(fixFemale(w.exp), fixMale(w.index), w.tpe, w.gender)
          case x => x map fixFemale
        }

        val sx = s match {
          case Connect(info, loc, exp) =>
            Connect(info, removeFemale(info, fixFemale(loc)), fixMale(exp))
          case (s) => s map (fixMale) map (onStmt)
        }
        stmts += sx
        if (stmts.size != 1) Block(stmts) else stmts(0)
      }
      Module(m.info, m.name, m.ports, onStmt(m.body))
    }
  
    val newModules = c.modules.map( _ match {
      case m: ExtModule => m
      case m: Module => remove_m(m)
    })
    Circuit(c.info, newModules, c.main)
  }
}
