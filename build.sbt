lazy val root = (project in file(".")).
  settings(
    name := "firrtl",
    version := "1.0",
    scalaVersion := "2.11.4"
  )

antlr4Settings

antlr4GenVisitor in Antlr4 := true // default = false

antlr4GenListener in Antlr4 := false // default = true

antlr4PackageName in Antlr4 := Option("firrtl.antlr")
