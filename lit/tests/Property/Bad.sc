// RUN: not scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s -- 2>&1 | FileCheck %s

import chisel3.properties._

class MyLit
// CHECK: error: unsupported Property type Bad.MyLit
val badPropLiteral = Property(new MyLit)

// CHECK: error: unsupported Property type Bad.MyTpe
class MyTpe
val badPropType = Property[MyTpe]()
