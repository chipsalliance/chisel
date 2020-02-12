resolvers += Resolver.url("scalasbt", new URL("https://scalasbt.artifactoryonline.com/scalasbt/sbt-plugin-releases")) (Resolver.ivyStylePatterns)

resolvers += Classpaths.sbtPluginReleases

resolvers += "jgit-repo" at "https://download.eclipse.org/jgit/maven"

addSbtPlugin("com.typesafe.sbt" % "sbt-site" % "1.4.0")

addSbtPlugin("com.eed3si9n" % "sbt-buildinfo" % "0.9.0")

addSbtPlugin("com.eed3si9n" % "sbt-unidoc" % "0.4.2")

addSbtPlugin("org.scalastyle" %% "scalastyle-sbt-plugin" % "1.0.0")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.6.1")

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.10")

addSbtPlugin("com.simplytyped" % "sbt-antlr4" % "0.8.1")

addSbtPlugin("com.github.gseitz" % "sbt-protobuf" % "0.6.5")

addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.9.11")

libraryDependencies += "com.github.os72" % "protoc-jar" % "3.11.1"
