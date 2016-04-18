/*
Copyright (c) 2014 - 2016 The Regents of the University of
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

package firrtl.passes

import firrtl._
import firrtl.Utils._
import firrtl.Mappers._

// Datastructures
import scala.collection.mutable.LinkedHashMap
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

object LowerTypes extends Pass {
   def name = "Lower Types"
   var mname = ""
   def is_ground (t:Type) : Boolean = {
      (t) match {
         case (_:UIntType|_:SIntType) => true
         case (t) => false
      }
   }
   def data (ex:Expression) : Boolean = {
      (kind(ex)) match {
         case (k:MemKind) => (ex) match {
            case (_:WRef|_:WSubIndex) => false
            case (ex:WSubField) => {
               var yes = ex.name match {
                  case "rdata" => true
                  case "data" => true
                  case "mask" => true
                  case _ => false
               }
               yes && ((ex.exp) match {
                  case (e:WSubField) => kind(e).as[MemKind].get.ports.contains(e.name) && (e.exp.typeof[WRef])
                  case (e) => false
               })
            }
            case (ex) => false
         }
         case (k) => false
      }
   }
   def expand_name (e:Expression) : Seq[String] = {
      val names = ArrayBuffer[String]()
      def expand_name_e (e:Expression) : Expression = {
         (e map (expand_name_e)) match {
            case (e:WRef) => names += e.name
            case (e:WSubField) => names += e.name
            case (e:WSubIndex) => names += e.value.toString
         }
         e
      }
      expand_name_e(e)
      names
   }
   def lower_other_mem (e:Expression, dt:Type) : Seq[Expression] = {
      val names = expand_name(e)
      if (names.size < 3) error("Shouldn't be here")
      create_exps(names(0),dt).map{ x => {
         var base = lowered_name(x)
         for (i <- 0 until names.size) {
            if (i >= 3) base = base + "_" + names(i)
         }
         val m = WRef(base, UnknownType(), kind(e), UNKNOWNGENDER)
         val p = WSubField(m,names(1),UnknownType(),UNKNOWNGENDER)
         WSubField(p,names(2),UnknownType(),UNKNOWNGENDER)
      }}
   }
   def lower_data_mem (e:Expression) : Expression = {
      val names = expand_name(e)
      if (names.size < 3) error("Shouldn't be here")
      else {
         var base = names(0)
         for (i <- 0 until names.size) {
            if (i >= 3) base = base + "_" + names(i)
         }
         val m = WRef(base, UnknownType(), kind(e), UNKNOWNGENDER)
         val p = WSubField(m,names(1),UnknownType(),UNKNOWNGENDER)
         WSubField(p,names(2),UnknownType(),UNKNOWNGENDER)
      }
   }
   def merge (a:String,b:String,x:String) : String = a + x + b
   def root_ref (e:Expression) : WRef = {
      (e) match {
         case (e:WRef) => e
         case (e:WSubField) => root_ref(e.exp)
         case (e:WSubIndex) => root_ref(e.exp)
         case (e:WSubAccess) => root_ref(e.exp)
      }
   }
   
   //;------------- Pass ------------------
   
   def lower_types (m:Module) : Module = {
      val mdt = LinkedHashMap[String,Type]()
      mname = m.name
      def lower_types (s:Stmt) : Stmt = {
         def lower_mem (e:Expression) : Seq[Expression] = {
            val names = expand_name(e)
            if (Seq("data","mask","rdata").contains(names(2))) Seq(lower_data_mem(e))
            else lower_other_mem(e,mdt(root_ref(e).name))
         }
         def lower_types_e (e:Expression) : Expression = {
            e match {
               case (_:WRef|_:UIntValue|_:SIntValue) => e
               case (_:WSubField|_:WSubIndex) => {
                  (kind(e)) match {
                     case (k:InstanceKind) => {
                        val names = expand_name(e)
                        var n = names(1)
                        for (i <- 0 until names.size) {
                           if (i > 1) n = n + "_" + names(i)
                        }
                        WSubField(root_ref(e),n,tpe(e),gender(e))
                     }
                     case (k:MemKind) => {
                        if (gender(e) != FEMALE) lower_mem(e)(0)
                        else e
                     }
                     case (k) => WRef(lowered_name(e),tpe(e),kind(e),gender(e))
                  }
               }
               case (e:DoPrim) => e map (lower_types_e)
               case (e:Mux) => e map (lower_types_e)
               case (e:ValidIf) => e map (lower_types_e)
            }
         }
         (s) match {
            case (s:DefWire) => {
               if (is_ground(s.tpe)) s else {
                  val es = create_exps(s.name,s.tpe)
                  val stmts = (es, 0 until es.size).zipped.map{ (e,i) => {
                     DefWire(s.info,lowered_name(e),tpe(e))
                  }}
                  Begin(stmts)
               }
            }
            case (s:DefPoison) => {
               if (is_ground(s.tpe)) s else {
                  val es = create_exps(s.name,s.tpe)
                  val stmts = (es, 0 until es.size).zipped.map{ (e,i) => {
                     DefPoison(s.info,lowered_name(e),tpe(e))
                  }}
                  Begin(stmts)
               }
            }
            case (s:DefRegister) => {
               if (is_ground(s.tpe)) s else {
                  val es = create_exps(s.name,s.tpe)
                  val inits = create_exps(s.init) 
                  val stmts = (es, 0 until es.size).zipped.map{ (e,i) => {
                     val init = lower_types_e(inits(i))
                     DefRegister(s.info,lowered_name(e),tpe(e),s.clock,s.reset,init)
                  }}
                  Begin(stmts)
               }
            }
            case (s:WDefInstance) => {
               val fieldsx = s.tpe.as[BundleType].get.fields.flatMap{ f => {
                  val es = create_exps(WRef(f.name,f.tpe,ExpKind(),times(f.flip,MALE)))
                  es.map{ e => {
                     gender(e) match {
                        case MALE => Field(lowered_name(e),DEFAULT,f.tpe)
                        case FEMALE => Field(lowered_name(e),REVERSE,f.tpe)
                     }
                  }}
               }}
               WDefInstance(s.info,s.name,s.module,BundleType(fieldsx))
            }
            case (s:DefMemory) => {
               mdt(s.name) = s.data_type
               if (is_ground(s.data_type)) s else {
                  val es = create_exps(s.name,s.data_type)
                  val stmts = es.map{ e => {
                     DefMemory(s.info,lowered_name(e),tpe(e),s.depth,s.write_latency,s.read_latency,s.readers,s.writers,s.readwriters)
                  }}
                  Begin(stmts)
               }
            }
            case (s:IsInvalid) => {
               val sx = (s map (lower_types_e)).as[IsInvalid].get
               kind(sx.exp) match {
                  case (k:MemKind) => {
                     val es = lower_mem(sx.exp)
                     Begin(es.map(e => {IsInvalid(sx.info,e)}))
                  }
                  case (_) => sx
               }
            }
            case (s:Connect) => {
               val sx = (s map (lower_types_e)).as[Connect].get
               kind(sx.loc) match {
                  case (k:MemKind) => {
                     val es = lower_mem(sx.loc)
                     Begin(es.map(e => {Connect(sx.info,e,sx.exp)}))
                  }
                  case (_) => sx
               }
            }
            case (s:DefNode) => {
               val locs = create_exps(s.name,tpe(s.value))
               val n = locs.size
               val nodes = ArrayBuffer[Stmt]()
               val exps = create_exps(s.value)
               for (i <- 0 until n) {
                  val locx = locs(i)
                  val expx = exps(i)
                  nodes += DefNode(s.info,lowered_name(locx),lower_types_e(expx))
               }
               if (n == 1) nodes(0) else Begin(nodes)
            }
            case (s) => s map (lower_types) map (lower_types_e)
         }
      }
   
      val portsx = m.ports.flatMap{ p => {
         val es = create_exps(WRef(p.name,p.tpe,PortKind(),to_gender(p.direction)))
         es.map(e => { Port(p.info,lowered_name(e),to_dir(gender(e)),tpe(e)) })
      }}
      (m) match {
         case (m:ExModule) => ExModule(m.info,m.name,portsx)
         case (m:InModule) => InModule(m.info,m.name,portsx,lower_types(m.body))
      }
   }
   
   def run (c:Circuit) : Circuit = {
      val modulesx = c.modules.map(m => lower_types(m))
      Circuit(c.info,modulesx,c.main)
   }
}

