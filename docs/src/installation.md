---
description: How to install and run Chisel
---

# Installation

Chisel is a [Scala](https://www.scala-lang.org/) library and compiler plugin.

## Quickstart with Scala CLI

The easiest way to install Chisel is to [install Scala CLI](https://scala-cli.virtuslab.org/install) to build and run the Chisel template:

```bash
wget https://github.com/chipsalliance/chisel/releases/latest/download/chisel-template.scala
scala-cli chisel-template.scala
```

The Chisel template is a simple, single-file example of Chisel that emits Verilog to the screen.

:::tip

While more complex projects often use a build tool like SBT or Mill as described below,
we still highly recommend using Scala CLI with the template for experimentation and
writing small snippets to share with others.

:::

## Dependencies

As described above, Scala CLI is a great "batteries included" way to use Chisel.
It will automatically download and manage all dependencies of Chisel _including a Java Development Kit (JDK)_.
More complex projects will require the user to install a JDK and a build tool.

:::info

Note that each of these dependencies are projects with their own installation instructions.
Please treat the commands below as suggestions and not directives.

:::

### Java Development Kit (JDK)

Scala runs on the Java Virtual Machine (JVM), so it is necessary to install a JDK to use Chisel.
Chisel works on any version of Java version 8 or newer; however, we recommend using an LTS release version 17 or newer.
Note that Scala CLI requires Java 17 or newer so unless your system installation of Java is at least version 17, Scala CLI will download Java 17 for its own use.

You can install any distribution of the JDK you prefer.
Eclipse Adoptium Temurin is a good option with support for all platforms: https://adoptium.net/installation

#### Ubuntu

Note that Temurin is not part of the default apt repositories, so you will need to add the Eclipse Adoptium repository.

Taken from the official [Temurin docs](https://adoptium.net/installation/linux/).
Please note that installation on most systems will require superuser priviledges (`sudo`).
You may need to prefix these commands with `sudo` including the `tee` commands following any pipes (`|`).

```sh
# Ensure the necessary packages are present:
apt install -y wget gpg apt-transport-https

# Download the Eclipse Adoptium GPG key:
wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null

# Configure the Eclipse Adoptium apt repository
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list

# Update the apt packages
apt update

# Install
apt install temurin-17-jdk
```

#### MacOS

Using [MacPorts](https://www.macports.org):
```sh
sudo port install openjdk17-temurin
```

Using [Homebrew](https://brew.sh):
```sh
brew tap homebrew/cask-versions
brew install --cask temurin17
```

#### Windows

Using [Scoop](https://scoop.sh):
```sh
scoop install temurin17-jdk
```

Windows users may also prefer using an [installer](https://adoptium.net/temurin/releases/?version=17&os=windows&arch=x86&package=jdk).


### Build Tools

Scala CLI is only intended for projects made up of a single to a handful of files.
Larger projects need a build tool.

#### Mill

[Mill](https://mill-build.com) is a modern Scala build tool with simple syntax and a better command-line experience than SBT.
We recommend Chisel users use Mill.

For detailed instructions, please see the [Mill documentation](https://mill-build.com/mill/Intro_to_Mill.html).

##### Linux and MacOS

The easiest way to use Mill is with the Mill Wrapper Script `millw`:

```sh
curl -L https://raw.githubusercontent.com/lefou/millw/0.4.11/millw > mill && chmod +x mill
```

You can then move this script to a global install location:

```sh
sudo mv mill /usr/local/bin/
```

##### Windows

Using [Scoop](https://scoop.sh):
```sh
scoop install mill
```

<!-- TODO flesh this out -->
Download `millw.bat`: https://raw.githubusercontent.com/lefou/millw/0.4.11/millw.bat.

#### SBT

[SBT](https://www.scala-sbt.org) is the more traditional Scala build tool that has many resources and examples.
It is most productively used with its own REPL rather than on the command-line.

##### Linux

The easiest way to install SBT is manually from the release tarball.
Note that MacOS and Windows users can also do a manual install.

```sh
curl -s -L https://github.com/sbt/sbt/releases/download/v1.9.8/sbt-1.9.8.tgz | tar xvz
```

Then copy the `sbt` bootstrap script into a global install location

```sh
sudo mv sbt/bin/sbt /usr/local/bin/
```

##### MacOS

Using [MacPorts](https://www.macports.org):
```sh
sudo port install sbt
```

##### Windows

Using [Scoop](https://scoop.sh):
```sh
scoop install sbt
```

### Firtool

Beginning with version 6.0, Chisel will manage the version of firtool on most systems.
However, it may be necessary to build from source on some systems (eg. older Linux distributions).
If you need to build firtool from source, please see the [Github repository](https://github.com/llvm/circt).

To see what version of firtool should be used for a given version of Chisel, see [Versioning](appendix/versioning#firtool-version).
