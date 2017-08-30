
resolvers ++= Seq(
  Resolver.sonatypeRepo("snapshots"),
  Resolver.sonatypeRepo("releases")
)

addSbtPlugin("edu.berkeley.cs" %% "sbt-chisel-dep" % "1.3-SNAPSHOT")
