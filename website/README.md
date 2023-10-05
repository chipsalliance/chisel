# Chisel-Lang Website

This directory provides the meta-website for the [Chisel](https://github.com/freechipsproject/chisel3) project.

# Contributing

We accept modifications to the website via Pull Requests.
All Pull Requests must both (1) be reviewed before they can be merged and (2) must pass Travis CI regression testing.
After a Pull Request is merged, a second Travis CI build will run on the `master` branch that will build and update the website.

## Requirements

To build the website you need:
* [`sbt`](https://www.scala-sbt.org/download.html)
* `jekyll`
* `gmake` - tested with version 4.2.1
* [`firtool`](https://github.com/llvm/circt/releases)

#### Installing jekyll

```
sudo apt-get install jekyll
gem install jekyll-redirect-from
```

## Building the Website

**tl;dr:**

``` bash
# Clone this git repository
git clone git@github.com:chipsalliance/chisel3

# Change into the directory where the clone lives
cd chisel3/website

# Build the website
make

# Serve the website
make serve

# In a web browser navigate to localhost:4000 to preview the website
```

The build process uses a [`Makefile`](https://github.com/chipsalliance/chisel3/blob/master/website/Makefile) to orchestrate building the website.
This `Makefile` does a number of actions:

#### 1. Determines Contributors

There have been a *lot* of contributors to Chisel, FIRRTL and associated projects.
As a small token of thanks, anyone who was contributed to these projects is [listed on the website's community tab](https://www.chisel-lang.org/community.html#contributors).

The website uses an `sbt` task that uses [`github4s`](https://github.com/47deg/github4s) to query GitHub for a list of contributors.

You can run this manually with:

```bash
make docs/src/main/tut/contributors.md
```

#### 2. Builds the Website

The actual website is assembled using [`sbt-microsite`](https://github.com/47deg/sbt-microsites).
You can build this manually with:

```bash
sbt docs/makeMicrosite
```

## Building and Testing Everything

To build the complete website use (and consider using the `-j` option with an appropriate number of parallel tasks to speed it up):

```bash
make
```

Initially building the website takes a long time (~45 minutes) due to the need to build Scaladoc documentation for versions.
However, this process is embarrassingly parallel and you only need to do it once.
All legacy/snapshot documentation will be cached in `$(buildDir)/api/`.
Due to this caching, building the website after changes takes only a couple of minutes (the website is big...).

After making modifications to the website, you can host it locally with (so long as you have installed `jekyll`):

```bash
make serve
```

Navigate to [`127.0.0.1:4000`](http://127.0.0.1:4000) to view your modifications to the website.

## Cleaning Things Up

There are two targets for cleaning the build:

- To clean the website build use `make clean` (*this will not remove built Scaladoc documentation*)
- To clean everything (including cached Scaladoc) use `make mrproper`

# Website Deployment

The Website is automatically deployed on every push to the `master` branch.
The built website is pushed to the `gh-pages` branch, which is the source
for what is hosted at https://chipsalliance.github.io/chisel3/.

Note that currently there is another website, https://www.chisel-lang.org/,
the source for which is https://github.com/freechipsproject/www.chisel-lang.org.
Following the 3.6 release, the chisel-lang.org URL will be changed to be sourced from this repository.