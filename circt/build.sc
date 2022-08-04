import coursier.maven.MavenRepository
import mill._
import mill.scalalib.TestModule.ScalaTest
import mill.scalalib._
import mill.scalalib.publish._
import mill.scalalib.scalafmt._

object v {
  val chiselCirct = "0.1"
  val chisel3 = ivy"edu.berkeley.cs::chisel3:3.5.0"
  val scalatest = ivy"org.scalatest::scalatest:3.2.7"
}

object chiselCirct extends mill.Cross[chiselCirctCrossModule]("2.12.12")

class chiselCirctCrossModule(val crossScalaVersion: String) extends CrossSbtModule with ScalafmtModule with PublishModule { m =>
  override def repositoriesTask = T.task {
    super.repositoriesTask() ++ Seq(
      MavenRepository("https://oss.sonatype.org/content/repositories/snapshots")
    )
  }
  
  override def millSourcePath = super.millSourcePath / os.up

  def publishVersion = v.chiselCirct

  def chisel3Module: Option[PublishModule] = None

  def chisel3IvyDeps = if (chisel3Module.isEmpty) Agg(v.chisel3) else Agg.empty[Dep]

  override def ivyDeps = super.ivyDeps() ++ chisel3IvyDeps

  override def moduleDeps = super.moduleDeps ++ chisel3Module
  
  object tests extends Tests with ScalaTest {
    override def ivyDeps = m.ivyDeps() ++ Agg(v.scalatest)
  }

  def pomSettings = PomSettings(
    description = "Infrastructure to compile Chisel projects using MLIR-based infrastructure (CIRCT)",
    organization = "com.sifive",
    url = "https://www.sifive.com/",
    licenses = Seq(License.`Apache-2.0`),
    versionControl = VersionControl.github("sifive", "chisel-circt"),
    developers = Seq(
      Developer("seldridge", "Schuyler Eldridge", "https://www.seldridge.dev")
    )
  )
}
