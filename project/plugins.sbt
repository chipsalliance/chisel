resolvers += Resolver.url("scalasbt", new URL("https://scalasbt.artifactoryonline.com/scalasbt/sbt-plugin-releases"))(
  Resolver.ivyStylePatterns
)

resolvers += Classpaths.sbtPluginReleases

resolvers += "jgit-repo".at("https://download.eclipse.org/jgit/maven")

addSbtPlugin("com.typesafe.sbt" % "sbt-site" % "1.4.0")

addSbtPlugin("com.eed3si9n" % "sbt-buildinfo" % "0.11.0")

addSbtPlugin("com.github.sbt" % "sbt-unidoc" % "0.5.0")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.9.3")

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "2.0.0")

addSbtPlugin("com.simplytyped" % "sbt-antlr4" % "0.8.2")

addSbtPlugin("com.github.sbt" % "sbt-protobuf" % "0.7.1")

addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.10.0")

addSbtPlugin("com.thoughtworks.sbt-api-mappings" % "sbt-api-mappings" % "3.0.2")

addSbtPlugin("org.scalameta" % "sbt-scalafmt" % "2.4.6")

addSbtPlugin("com.typesafe" % "sbt-mima-plugin" % "1.1.0")

addSbtPlugin("com.github.sbt" % "sbt-ci-release" % "1.5.10")

libraryDependencies += "com.github.os72" % "protoc-jar" % "3.11.4"
