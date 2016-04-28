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
import scala.collection.mutable.ArrayBuffer

object CheckInitialization extends Pass with StanzaPass {
   def name = "Check Initialization"
   var mname = ""
   class RefNotInitialized (info:Info, name:String) extends PassException(s"${info} : [module ${mname}  Reference ${name} is not fully initialized.")
   def run (c:Circuit): Circuit = {
      val errors = ArrayBuffer[PassException]()
      def check_init_m (m:InModule) : Unit = {
         def get_name (e:Expression) : String = {
            (e) match { 
               case (e:WRef) => e.name
               case (e:WSubField) => get_name(e.exp) + "." + e.name
               case (e:WSubIndex) => get_name(e.exp) + "[" + e.value + "]"
               case (e) => error("Shouldn't be here"); ""
            }
         }
         def has_voidQ (e:Expression) : Boolean = {
            var void = false
            def has_void (e:Expression) : Expression = {
               (e) match { 
                  case (e:WVoid) => void = true; e
                  case (e) => e map (has_void)
               }
            }
            has_void(e)
            void
         }
         def check_init_s (s:Stmt) : Stmt = {
            (s) match { 
               case (s:Connect) => {
                  if (has_voidQ(s.exp)) errors += new RefNotInitialized(s.info,get_name(s.loc))
                  s
               }
               case (s) => s map (check_init_s)
            }
         }
         check_init_s(m.body)
      }
         
      for (m <- c.modules) {
         mname = m.name
         (m) match  { 
            case (m:InModule) => check_init_m(m)
            case (m) => false
         }
      }

    if (errors.nonEmpty) throw new PassExceptions(errors)
    c
   }
}
