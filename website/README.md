# Website

This website is built using [Docusaurus 3](https://docusaurus.io/), a modern static website generator built in JavaScript using React.

As the website depends on information and documentation compiled using sbt, we have wrapped the calls to sbt and npm with a Makefile in this directory.
It is difficult to properly capture dependencies handled by sbt or npm in Make, so this Makefile is likely to be buggy.
If things are not behaving as expected, make sure to run `make clean` or `make mrproper`.


## Dependencies

Building the website requires the usual [Chisel Setup](../SETUP.md), in addition to Node.js and npm.

Please follow the [official npm instructions](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) for installing Node.js and npm.

In the likely event that these instructions become out-of-date, please see the [Github Actions workflows](../.github/workflows) for how the website is tested and deployed in CI.

## Installation

Before running anything else you need to install all website dependencies.

```
make install
```

## Build

```
make build
```

This command generates the static website into the `build` directory.
It has many steps handled by the [Makefile](./Makefile):

1. Compile the Chisel Scala source code
2. Run mdoc to generate markdown for the website
3. Copy the generaeted markdown into `docs/`
4. Determine contributors to the Chisel project and generate `src/pages/generated/contributors.md`
5. Run docusaurus to generate the website

## Development

You can view the built website locally with:

```
make serve
```

This will locally host the website so that you can see the impact of any local changes.

## Deployment

Deployment is handled automatically upon push to main by CI.

