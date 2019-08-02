addSbtPlugin("com.eed3si9n" % "sbt-unidoc" % "0.4.1")
addSbtPlugin("com.47deg"  % "sbt-microsites" % "0.9.1")
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.6")
addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.9.4")
addSbtPlugin("com.github.gseitz" % "sbt-protobuf" % "0.6.3")
addSbtPlugin("com.simplytyped" % "sbt-antlr4" % "0.8.1")
addSbtPlugin("org.scalastyle" %% "scalastyle-sbt-plugin" % "1.0.0")
addSbtPlugin("com.eed3si9n" % "sbt-buildinfo" % "0.7.0")
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.5.1")

libraryDependencies += "com.github.os72" % "protoc-jar" % "3.5.1.1"
