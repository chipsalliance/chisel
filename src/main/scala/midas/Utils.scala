
package midas

import firrtl._

object Utils {

  // Takes a set of strings or ints and returns equivalent expression node
  //   Strings correspond to subfields/references, ints correspond to indexes
  // eg. Seq(io, port, ready)    => io.port.ready
  //     Seq(io, port, 5, valid) => io.port[5].valid
  //     Seq(3)                  => UInt("h3")
  def buildExp(names: Seq[Any]): Exp = {
    def rec(names: Seq[Any]): Exp = {
      names.head match {
        // Useful for adding on indexes or subfields
        case head: Exp => head 
        // Int -> UInt/SInt/Index
        case head: Int => 
          if( names.tail.isEmpty ) // Is the UInt/SInt inference good enough?
            if( head > 0 ) UIntValue(head, UnknownWidth) else SIntValue(head, UnknownWidth)
          else Index(rec(names.tail), head, UnknownType)
        // String -> Ref/Subfield
        case head: String => 
          if( names.tail.isEmpty ) Ref(head, UnknownType)
          else Subfield(rec(names.tail), head, UnknownType)
        case _ => throw new Exception("Invalid argument type to buildExp! " + names)
      }
    }
    rec(names.reverse) // Let user specify in more natural format
  }
  def buildExp(name: Any): Exp = buildExp(Seq(name))

  def genPrimopReduce(op: Primop, args: Seq[Exp]): DoPrimop = {
    if( args.length == 2 ) DoPrimop(op, Seq(args.head, args.last), Seq(), UnknownType)
    else DoPrimop(op, Seq(args.head, genPrimopReduce(op, args.tail)), Seq(), UnknownType)
  }

  // For a port that is known to be of type BundleType, return the fields of that bundle
  def getFields(port: Port): Seq[Field] = {
    port.tpe match {
      case b: BundleType => b.fields
      case _ => throw new Exception("getFields called on invalid port " + port)
    }
  }

}
