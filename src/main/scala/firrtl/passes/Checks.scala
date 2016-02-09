
package firrtl.passes

import com.typesafe.scalalogging.LazyLogging

// Datastructures
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

import firrtl._
import firrtl.Utils._
import firrtl.PrimOps._

object CheckHighForm extends Pass with LazyLogging {
  def name = "High Form Check"

  // Custom Exceptions
  class NotUniqueException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Reference ${name} does not have a unique name.")
  class IsPrefixException(prefix: String) extends PassException(s"${sinfo}: [module ${mname}] Symbol ${prefix} is a prefix.")
  class InvalidLOCException extends PassException(s"${sinfo}: [module ${mname}] Invalid connect to an expression that is not a reference or a WritePort.")
  class NegUIntException extends PassException(s"${sinfo}: [module ${mname}] UIntValue cannot be negative.")
  class UndeclaredReferenceException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Reference ${name} is not declared.")
  class PoisonWithFlipException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Poison ${name} cannot be a bundle type with flips.")
  class MemWithFlipException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Memory ${name} cannot be a bundle type with flips.")
  class InvalidAccessException extends PassException(s"${sinfo}: [module ${mname}] Invalid access to non-reference.")
  class NoTopModuleException(name: String) extends PassException(s"${sinfo}: A single module must be named ${name}.")
  class ModuleNotDefinedException(name: String) extends PassException(s"${sinfo}: Module ${name} is not defined.")
  class IncorrectNumArgsException(op: String, n: Int) extends PassException(s"${sinfo}: [module ${mname}] Primop ${op} requires ${n} expression arguments.")
  class IncorrectNumConstsException(op: String, n: Int) extends PassException(s"${sinfo}: [module ${mname}] Primop ${op} requires ${n} integer arguments.")
  class NegWidthException extends PassException(s"${sinfo}: [module ${mname}] Width cannot be negative or zero.")
  class NegVecSizeException extends PassException(s"${sinfo}: [module ${mname}] Vector type size cannot be negative.")
  class NegMemSizeException extends PassException(s"${sinfo}: [module ${mname}] Memory size cannot be negative or zero.")
  // Note the following awkward strings are due to an issue with Scala string interpolation and escaped double quotes
  class BadPrintfException(x: Char) extends PassException(s"${sinfo}: [module ${mname}] Bad printf format: " + "\"%" + x + "\"")
  class BadPrintfTrailingException extends PassException(s"${sinfo}: [module ${mname}] Bad printf format: trailing " + "\"%\"")
  class BadPrintfIncorrectNumException extends PassException(s"${sinfo}: [module ${mname}] Bad printf format: incorrect number of arguments")

  // Trie Datastructure for prefix checking
  case class Trie(var children: HashMap[String, Trie], var end: Boolean) {
    def empty: Boolean = children.isEmpty
    def add(ls: Seq[String]): Boolean = {
      var t: Trie = this
      var sawEnd = false
      for (x <- ls) {
        if (t.end) sawEnd = true
        if (t.contains(x)) t = t.children(x)
        else {
          val temp = new Trie(HashMap[String,Trie](),false)
          t.children(x) = temp
          t = temp
        }
      }
      t.end = true
      sawEnd | !t.empty
    }
    def contains(s: String): Boolean = children.contains(s)
    def contains(ls: Seq[String]): Boolean = {
      var t: Trie = this
      for (x <- ls) {
        if (t.contains(x)) t = t.children(x)
        else return false
      }
      t.end
    }
  }

  // Utility functions
  def hasFlip(t: Type): Boolean = {
    var has = false
    def findFlip(t: Type): Type = {
      t match {
        case t: BundleType => {
          for (f <- t.fields) {
            if (f.flip == REVERSE) has = true
          }
          t
        }
        case t: Type => t
      }
    }
    findFlip(t)
    tMap(findFlip _, t)
    has
  }

  // TODO FIXME
  // - Do we need to check for uniquness on port names?
  // Global Variables
  private var mname: String = ""
  private var sinfo: Info = NoInfo
  def run (c:Circuit): Circuit = {
    val errors = ArrayBuffer[PassException]()
    def checkHighFormPrimop(e: DoPrim) = {
      def correctNum(ne: Option[Int], nc: Int) = {
        ne match {
          case Some(i) => if(e.args.length != i) errors.append(new IncorrectNumArgsException(e.op.getString, i))
          case None => // Do Nothing
        }
        if (e.consts.length != nc) errors.append(new IncorrectNumConstsException(e.op.getString, nc))
      }

      e.op match {
        case ADD_OP             => correctNum(Option(2),0)
        case SUB_OP             => correctNum(Option(2),0)
        case MUL_OP             => correctNum(Option(2),0)
        case DIV_OP             => correctNum(Option(2),0)
        case REM_OP             => correctNum(Option(2),0)
        case LESS_OP            => correctNum(Option(2),0)
        case LESS_EQ_OP         => correctNum(Option(2),0)
        case GREATER_OP         => correctNum(Option(2),0)
        case GREATER_EQ_OP      => correctNum(Option(2),0)
        case EQUAL_OP           => correctNum(Option(2),0)
        case NEQUAL_OP          => correctNum(Option(2),0)
        case PAD_OP             => correctNum(Option(1),1)
        case AS_UINT_OP         => correctNum(Option(1),0)
        case AS_SINT_OP         => correctNum(Option(1),0)
        case AS_CLOCK_OP        => correctNum(Option(1),0)
        case SHIFT_LEFT_OP      => correctNum(Option(1),1)
        case SHIFT_RIGHT_OP     => correctNum(Option(1),1)
        case DYN_SHIFT_LEFT_OP  => correctNum(Option(2),0)
        case DYN_SHIFT_RIGHT_OP => correctNum(Option(2),0)
        case CONVERT_OP         => correctNum(Option(1),0)
        case NEG_OP             => correctNum(Option(1),0)
        case NOT_OP             => correctNum(Option(1),0)
        case AND_OP             => correctNum(Option(2),0)
        case OR_OP              => correctNum(Option(2),0)
        case XOR_OP             => correctNum(Option(2),0)
        case AND_REDUCE_OP      => correctNum(None,0)
        case OR_REDUCE_OP       => correctNum(None,0)
        case XOR_REDUCE_OP      => correctNum(None,0)
        case CONCAT_OP          => correctNum(Option(2),0)
        case BITS_SELECT_OP     => correctNum(Option(1),2)
        case HEAD_OP            => correctNum(Option(1),1)
        case TAIL_OP            => correctNum(Option(1),1)
      }
    }

    def checkFstring(s: String, i: Int) = {
      val validFormats = "bedxs"
      var percent = false
      var ret = true
      var npercents = 0
      for (x <- s) { 
        if (!validFormats.contains(x) && percent)
          errors.append(new BadPrintfException(x))
        if (x == '%') npercents = npercents + 1
        percent = (x == '%')
      }
      if (percent) errors.append(new BadPrintfTrailingException)
      if (npercents != i) errors.append(new BadPrintfIncorrectNumException)
    }
    def checkValidLoc(e: Expression) = {
      e match {
        case e @ ( _: UIntValue | _: SIntValue | _: DoPrim ) => errors.append(new InvalidLOCException)
        case _ => // Do Nothing
      }
    }
    def checkHighFormW(w: Width): Width = {
      w match {
        case w: IntWidth => 
          if (w.width <= BigInt(0)) errors.append(new NegWidthException)
        case _ => // Do Nothing
      }
      w
    }
    def checkHighFormT(t: Type): Type = {
      tMap(checkHighFormT _, t) match {
        case t: VectorType => 
          if (t.size < 0) errors.append(new NegVecSizeException)
        case _ => // Do nothing
      }
      wMap(checkHighFormW _, t)
    }

    def checkHighFormM(m: Module): Module = {
      val names = HashMap[String, Boolean]()
      val mnames = HashMap[String, Boolean]()
      val tries = Trie(HashMap[String, Trie](),false)
      def checkHighFormE(e: Expression): Expression = {
        def validSubexp(e: Expression): Expression = {
          e match {
            case e @ (_: WRef | _: WSubField | _: WSubIndex | _: WSubAccess | _: Mux | _: ValidIf) => // No error
            case _ => errors.append(new InvalidAccessException)
          }
          e
        }
        eMap(checkHighFormE _, e) match {
          case e: WRef => 
            if (!names.contains(e.name)) errors.append(new UndeclaredReferenceException(e.name))
          case e: DoPrim => checkHighFormPrimop(e)
          case e: WSubAccess => {
            validSubexp(e.exp)
            e
          }
          case e: UIntValue => 
            if (e.value < 0) errors.append(new NegUIntException)
          case e => eMap(validSubexp _, e)
        }
        wMap(checkHighFormW _, e)
        tMap(checkHighFormT _, e)
        e
      }
      def checkHighFormS(s: Stmt): Stmt = {
        def checkName(name: String): String = {
          if (names.contains(name)) errors.append(new NotUniqueException(name))
          else names(name) = true
          val ls: Seq[String] = name.split('$')
          if (tries.add(ls)) errors.append(new IsPrefixException(name))
          name 
        }
        sinfo = s.getInfo

        stMap(checkName _, s)
        tMap(checkHighFormT _, s)
        eMap(checkHighFormE _, s)
        s match {
          case s: DefPoison => {
             if (hasFlip(s.tpe)) errors.append(new PoisonWithFlipException(s.name))
             checkHighFormT(s.tpe)
          }
          case s: DefMemory => { 
            if (hasFlip(s.data_type)) errors.append(new MemWithFlipException(s.name))
            if (s.depth <= 0) errors.append(new NegMemSizeException)
          }
          case s: WDefInstance => { 
            if (!c.modules.map(_.name).contains(s.module))
              errors.append(new ModuleNotDefinedException(s.module))
          }
          case s: Connect => checkValidLoc(s.loc)
          case s: BulkConnect => checkValidLoc(s.loc)
          case s: Print => checkFstring(s.string, s.args.length)
          case _ => // Do Nothing
        }

        sMap(checkHighFormS _, s)
      }

      mname = m.name
      for (m <- c.modules) {
        mnames(m.name) = true
      }
      for (p <- m.ports) {
        // FIXME should we set sinfo here?
        names(p.name) = true
        val tpe = p.getType
        tMap(checkHighFormT _, tpe)
        wMap(checkHighFormW _, tpe)
      }

      m match {
        case m: InModule => checkHighFormS(m.body)
        case m: ExModule => // Do Nothing
      }
      m
    }
    
    var numTopM = 0
    for (m <- c.modules) {
      if (m.name == c.main) numTopM = numTopM + 1
      checkHighFormM(m)
    }
    sinfo = c.info
    if (numTopM != 1) errors.append(new NoTopModuleException(c.main))
    if (errors.nonEmpty) throw new PassExceptions(errors)
    c
  }
}

