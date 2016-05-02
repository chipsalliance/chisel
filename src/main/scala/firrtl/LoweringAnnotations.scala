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

package firrtl

// ===========================================
//            Lowering Annotations
// -------------------------------------------

case object TagAnnotation extends Annotation {
   def serialize: String = this.toString
}
case class StringAnnotation (string: String) extends Annotation {
   def serialize: String = string
}

// Annotated names cannot be renamed, split, or removed
case class BrittleCircuitAnnotation(kind: CircuitAnnotationKind, map: Map[Named,Annotation]) extends CircuitAnnotation {
   def check(renames: RenameMap): Unit = {
      map.keys.foreach { mn => {
         renames.map.get(mn) match {
            case None => {}
            case Some(rename) => rename.size match {
               case 1 if (rename.head.name == mn.name) => {} // Ok case
               case 0 => throw new AnnotationException(s"Cannot remove $mn.")
               case 1 => throw new AnnotationException(s"Cannot rename $mn into ${rename.head.name}.")
               case _ =>
                  throw new AnnotationException(s"Cannot split ${mn.name}into " + rename.map(_.name).reduce(_ + " and " + _) + ".")
            }
         }
      }}
   }
   def update(renames: RenameMap): BrittleCircuitAnnotation = {
      check(renames)
      BrittleCircuitAnnotation(kind, map)
   }
}

// Annotation moves with renamed and split names. Removed annotated names do not trigger an error.
case class StickyCircuitAnnotation(kind: CircuitAnnotationKind, map: Map[Named,Annotation]) extends CircuitAnnotation {
   def get(name: Named): Annotation = map(name)
   def update(renames: RenameMap): StickyCircuitAnnotation = {
      val mapx = renames.map.foldLeft(Map[Named, Annotation]()){
         case (newmap: Map[Named, Annotation],(name: Named, renames: Seq[Named])) => {
            if (map.contains(name)) {
               renames.map(rename => Map(rename -> map(name))).reduce(_ ++ _) ++ newmap
            } else newmap
         }
      }
      StickyCircuitAnnotation(kind, mapx)
   }
   def getModuleNames: Seq[ModuleName] = map.keys.flatMap{
      _ match {
         case x: ModuleName => Seq(x)
         case _ => Seq.empty
      }
   }.toSeq
   def getComponentNames: Seq[ComponentName] = map.keys.flatMap{
      _ match {
         case x: ComponentName => Seq(x)
         case _ => Seq.empty
      }
   }.toSeq
}
