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

/* Unfinished. Has 3 basic parameters available */
package Chisel

import Builder._
// import Node._
import Module._
import JHFormat._

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

import java.lang.reflect.{Type, ParameterizedType}

import scala.io.Source
import java.io._

//>Params.scala: Implementation of parameter framework. Defines case class
  //containers for parameter types. Params object is what actually stores
  //the data structures of parameters, whether they are generated from a Chisel
  //design, or read from a json file

case class ParamInvalidException(msg: String) extends Exception

abstract class Param[+T] {
  def init: T
  def max: Int
  def min: Int
  def pname: String
  var index: Int = -2
  var gID: Int = -1
  var register = Params.register(getComponent(), pname, this)
  //def register(pName: String) = { pname = pName; Params.register(jack.getComponent(), pname, this)}
  def getValue: T = Params.getValue(getComponent(),this.pname,this).asInstanceOf[T]
}

case class ValueParam(pname:String, init: Any) extends Param[Any] {
  val max = init.toString.toInt
  val min = init.toString.toInt
}

case class RangeParam(pname:String, init: Int, min: Int, max: Int) extends Param[Int]

case class LessParam(pname:String, init: Int, min: Int, par: Param[Any]) extends Param[Int] {
  val max = par.max
}

case class LessEqParam(pname:String, init: Int, min: Int, par: Param[Any]) extends Param[Int] {
  val max = par.max
}

case class GreaterParam(pname:String, init: Int, par: Param[Any], max: Int) extends Param[Int] {
  val min = par.min
}

case class GreaterEqParam(pname:String, init: Int, par: Param[Any], max: Int) extends Param[Int] {
  val min = par.min
}

case class DivisorParam(pname:String, init: Int, min: Int, max: Int, par: Param[Any]) extends Param[Int]

case class EnumParam(pname:String, init: String, values: List[String]) extends Param[String] {
  val max = init.toString.toInt
  val min = init.toString.toInt
}

object IntParam {
  def apply(name: String, init: Int) = RangeParam(name, init, init, init)
}

object Params {
  type Space = JHFormat.Space
  var space = new Space
  var design = new Space
  var modules = new HashMap[String, Module]
  var gID: Int = 0

  var buildingSpace = true

  def getValue(module: Module, pname: String, p: Param[Any]) = {
    val mname= if(module == null) "TOP" else {module.getClass.getName}
    if(buildingSpace) p.init
    else{
      val x = design.find(t => (t._3) == (p.gID))
      if(x.isEmpty){
        throw new ParamInvalidException("Missing parameter " + pname + " in Module " + mname)
      } else {
        x.get._2.init
      }
    }
  }

  def register(module: Module, pname: String, p: Param[Any]) = {
    val mname= if(module == null) "TOP" else {module.getClass.getName}
    modules(mname) = module
    if(buildingSpace) {
      space += ((mname,p,gID))
    }
    p.gID = gID
    gID += 1
    p
  }

  def dump_file(filename: String, design: Space) = {
    val string = JHFormat.serialize(design)
    val writer = new PrintWriter(new File(filename))
    println("Dumping to " + filename + ":\n" + string)
    writer.write(string)
    writer.close()
  }

  def dump(dir: String) = {
    buildingSpace = false
    dump_file(dir + "/" + JHFormat.spaceName, Params.space)
  }
  def load(designName: String) = {
    buildingSpace = false
    design = JHFormat.deserialize(designName)
    gID = 0
  }

  def toCxxStringParams : String = {
    var string = new StringBuilder("")
    for ((mname, p, gID) <- space) {
      val rmname = if (mname == "TOP") "" else modules(mname).name + "__";
      string ++= "const int " + rmname + p.pname + " = " + toCxxStringParam(p) + ";\n"
    }
    string.toString
  }

  def toDotpStringParams : String = {
    var string = new StringBuilder("")
    for ((mname, p, gID) <- space) {
      val rmname = if (mname == "TOP") "" else modules(mname).name + ":";
      string ++= rmname + p.pname + " = " + toCxxStringParam(p) + "\n"
    }
    string.toString
  }


  def toCxxStringParam(param: Param[Any]) = {
    param match {
      // case EnumParam(init, list) =>
        //"(range," + init + "," + list + ")"
      //   "const int " + name + " = " + init + ";\n"
      case ValueParam(pname, init) =>
        init.toString
      case RangeParam(pname, init, min, max) =>
        init.toString
      case LessParam(pname, init, min, par) =>
        init.toString
      case LessEqParam(pname, init, min, par) =>
        init.toString
      case GreaterParam(pname, init, min, par) =>
        init.toString
      case GreaterEqParam(pname, init, min, par) =>
        init.toString
      case DivisorParam(pname, init, min, max, par) =>
        init.toString
      case EnumParam(pname, init, values) =>
        init.toString
      case _ =>
        throw new ParamInvalidException("Unknown parameter class!"); ""
    }
  }
}
