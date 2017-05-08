package chiselBuild

import sbt._
import Keys._

// If you make changes here, you may need to update the SubprojectGenerator.
object ChiselSettings {

  lazy val commonSettings = Seq (
    organization := "edu.berkeley.cs",
    scalaVersion := "2.11.8",

    resolvers ++= Seq(
      Resolver.sonatypeRepo("snapshots"),
      Resolver.sonatypeRepo("releases")
    ),

    javacOptions ++= Seq("-source", "1.7", "-target", "1.7")
  )

  lazy val publishSettings = Seq (
    publishMavenStyle := true,
    publishArtifact in Test := false,
    pomIncludeRepository := { x => false },

    publishTo <<= version { v: String =>
      val nexus = "https://oss.sonatype.org/"
      if (v.trim.endsWith("SNAPSHOT")) {
        Some("snapshots" at nexus + "content/repositories/snapshots")
      }
      else {
        Some("releases" at nexus + "service/local/staging/deploy/maven2")
      }
    }
  )
}
