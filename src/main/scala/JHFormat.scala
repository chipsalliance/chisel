/*
 Copyright (c) 2011, 2012, 2013, 2014 The Regents of the University of
 California (Regents). All Rights Reserved.  Redistribution and use in
 source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:

    * Redistributions of source code must retain the above
      copyright notice, this list of conditions and the following
      two paragraphs of disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      two paragraphs of disclaimer in the documentation and/or other materials
      provided with the distribution.
    * Neither the name of the Regents nor the names of its contributors
      may be used to endorse or promote products derived from this
      software without specific prior written permission.

 IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
 REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
 ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
 TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 MODIFICATIONS.
*/

package Chisel
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io._

object JHFormat {
  type Space = ArrayBuffer[(String,Param[Any],Int)]
  def serialize[T<:Param[Any]](space: Space) : String = {
    var string = new StringBuilder("")
    for ((mname, p, gID) <- space) {
      string ++= mname + "," + toStringParam(p) + "\n"
    }
    string.toString
  }

  def deserialize(filename: String):Space = {
    var lines = io.Source.fromFile(filename).getLines
    var space = new Space
    while(lines.hasNext) {
      val line = lines.next()
      val args = line.split(",")
      val mname = args(0)
      val ptype = args(1)
      val gID   = args(2).toInt
      val pname = args(3)
      val param = ptype match {
        case "value"   => { val p = new ValueParam(pname,args(4).toInt)
          p.gID = gID; p }
        case "range"   => { val p = new RangeParam(pname,args(4).toInt,args(5).toInt,args(6).toInt)
          p.gID = gID; p }
        case "less"    => { val p = new LessParam(pname,args(4).toInt,args(5).toInt,space.find(i => i._3 == args(6).toInt).get._2)
          p.gID = gID; p }
        case "lesseq"  => { val p = new LessEqParam(pname,args(4).toInt,args(5).toInt,space.find(i => i._3 == args(6).toInt).get._2)
          p.gID = gID; p }
        case "great"   => { val p = new GreaterParam(pname,args(4).toInt,space.find(i => i._3 == args(5).toInt).get._2,args(6).toInt)
          p.gID = gID; p }
        case "greateq" => { val p = new GreaterEqParam(pname,args(4).toInt,space.find(i => i._3 == args(5).toInt).get._2,args(6).toInt)
          p.gID = gID; p }
        case "divisor" => { val p = new DivisorParam(pname,args(4).toInt,args(5).toInt,args(6).toInt,space.find(i => i._3 == args(7).toInt).get._2)
          p.gID = gID; p }
        case "enum"    => { val p = new EnumParam(pname,args(4),args.slice(5,args.length).toList)
          p.gID = gID; p }
        case _         => { throw new ParamInvalidException("Unknown parameter"); new ValueParam("error",0) }
      }
      space += ((mname,param,gID.toInt))
    }
    space
  }

  def toStringParam(param: Param[Any]):String = {
    param match {
      case ValueParam(pname, init) =>
        "value,"   + param.gID + "," + pname + "," + init
      case RangeParam(pname, init, min, max) =>
        "range,"   + param.gID + "," + pname + "," + init + "," + min + "," + max
      case LessParam(pname, init, min, par) =>
        "less,"    + param.gID + "," + pname + "," + init + "," + min + "," + par.gID
      case LessEqParam(pname, init, min, par) =>
        "lesseq,"  + param.gID + "," + pname + "," + init + "," + min + "," + par.gID
      case GreaterParam(pname, init, par, max) =>
        "great,"   + param.gID + "," + pname + "," + init + "," + par.gID + "," + max
      case GreaterEqParam(pname, init, par, max) =>
        "greateq," + param.gID + "," + pname + "," + init + "," + par.gID + "," + max
      case DivisorParam(pname, init, min, max, par) =>
        "divisor," + param.gID + "," + pname + "," + init + "," + min + "," + max + "," + par.gID
      case EnumParam(pname, init, values) =>
        "enum,"    + param.gID + "," + pname + "," + init + "," + values.mkString(",")
      case _ =>
        throw new ParamInvalidException("Unknown parameter class!"); ""
    }
  }

  val spaceName = "space.prm"
}
