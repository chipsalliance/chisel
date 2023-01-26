---
layout: page
title:  "Community"
section: "Community"
position: 16
redirect_from: /documentation
---

## Chisel Users Community

If you're a Chisel user and want to stay connected to the wider user community, any of the following are great avenues:

- Interact with other Chisel users in one of our Gitter chat rooms:
  - [Chisel](https://gitter.im/freechipsproject/chisel3)
  - [FIRRTL](https://gitter.im/freechipsproject/firrtl)
- Ask/Answer [Questions on Stack Overflow using the `[chisel]` tag](https://stackoverflow.com/questions/tagged/chisel)
- Ask questions and discuss ideas on the Chisel/FIRRTL Mailing Lists:
  - [Chisel Users](https://groups.google.com/forum/#!forum/chisel-users)
  - [Chisel Developers](https://groups.google.com/forum/#!forum/chisel-dev)
- Follow us on our [`@chisel_lang` Twitter Account](https://twitter.com/chisel_lang)
- Subscribe to our [`chisel-lang` YouTube Channel](https://www.youtube.com/c/chisel-lang)

## Projects Using Chisel/FIRRTL

If you want to add your project to the list, let us know on the [Chisel users mailing list](http://groups.google.com/group/chisel-users)!

### Chisel

| Project                                                                                                                                      | Description                                                     | Author                                                                           | Links                                                                                                        |
|----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| [Rocket Chip Generator](https://github.com/chipsalliance/rocket-chip)                                                                        | RISC-V System-on-Chip Generator, 5-stage RISC-V Microprocessor  | [`@ucb-bar`](https://github.com/ucb-bar), [`@sifive`](https://github.com/sifive) | [Report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-17.html)                                |
| [Berkeley Out-of-order Machine](https://github.com/ucb-bar/riscv-boom)                                                                       | RISC-V Out-of-order/Multi-issue Microprocessor                  | [`@ucb-bar`](https://github.com/ucb-bar)                                         | [Site](https://boom-core.org/), [Thesis](http://www2.eecs.berkeley.edu/Pubs/TechRpts/2018/EECS-2018-151.pdf) |
| [RISC-V Mini](https://github.com/ucb-bar/riscv-mini)                                                                                         | 3-stage RISC-V Microprocessor                                   | [`@ucb-bar`](https://github.com/ucb-bar)                                         |                                                                                                              |
| [Sodor Processor Collection](https://github.com/ucb-bar/riscv-sodor)                                                                         | Educational RISC-V Microprocessors (1, 2, 3, 5-stage)           | [`@ucb-bar`](https://github.com/ucb-bar)                                         |                                                                                                              |
| [Patmos](https://github.com/t-crest/patmos)                                                                                                  | Time-predictable VLIW processor                                 | [`@t-crest`](https://github.com/t-crest)                                         | [Site](http://patmos.compute.dtu.dk/)                                                                        |
| [OpenSoC Fabric](https://github.com/LBL-CoDEx/OpenSoCFabric)                                                                                 | Parametrizable Network-on-Chip Generator                        | [`@LBL-CoDEx`](https://github.com/LBL-CoDEx)                                     | [Site](http://www.opensocfabric.org)                                                                         |
| [Hwacha](https://github.com/ucb-bar/hwacha)                                                                                                  | Decoupled Vector-fetch Accelerator                              | [`@ucb-bar`](https://github.com/ucb-bar)                                         | [Report](https://people.eecs.berkeley.edu/~krste/papers/EECS-2015-263.pdf)                                   |
| [DANA](https://github.com/bu-icsg/dana)                                                                                                      | Multilayer Perceptron Accelerator for Rocket                    | [`@bu-icsg`](https://github.com/bu-icsg)                                         | [Paper](http://people.bu.edu/schuye/files/pact2015-eldridge-paper.pdf)                                       |
| [Gemmini](https://github.com/ucb-bar/gemmini)                                                                                                | Systolic-array Accelerator Generator                            | [`@ucb-bar`](https://github.com/ucb-bar)                                         | [Paper](https://arxiv.org/pdf/1911.09925)                                                                    |
| [Edge TPU](https://cloud.google.com/edge-tpu)                                                                                                | AI Inference Accelerator                                        | [`@google`](https://github.com/google)                                           | [Video](https://www.youtube.com/watch?v=x85342Cny8c)                                                         |
| [ChiselFlow](https://github.com/apl-cornell/ChiselFlow)                                                                                      | Information Flow Types in Chisel3                               | [`@apl-cornell`](https://github.com/apl-cornell)                                 | [Paper](https://ecommons.cornell.edu/xmlui/bitstream/handle/1813/57673/paper.pdf)                            |
| [PHMon](https://github.com/bu-icsg/PHMon/tree/code)                                                                                          | Programmable Hardware Monitor                                   | [`@bu-icsg`](https://github.com/bu-icsg)                                         | [Paper](http://people.bu.edu/joshi/files/sec20spring_delshadtehrani_prepub.pdf)                              |
| [DINO CPU](https://github.com/jlpteaching/dinocpu)                                                                                           | Davis In-Order (DINO) CPU models                                | [`@jlpteaching`](https://github.com/jlpteaching)                                 | [Paper](https://dl.acm.org/doi/10.1145/3338698.3338892)                                                      |
| [Quasar](https://github.com/Lampro-Mellon/Quasar)                                                                                            | CHISEL implementation of SweRV-EL2                              | [`@Lampro-Mellon`](https://github.com/Lampro-Mellon)                             | [Video](https://www.youtube.com/watch?v=R9eCNmGa5Vc)                                                         |
| FP Divider [Pipelined](https://github.com/Ssavas/fp-division-pipelined) / [Not Pipelined](https://github.com/Ssavas/fp-division-no-pipeline) | IEEE binary 32-bit divider using Harmonized Parabolic Synthesis | [`@Ssavas`](https://github.com/Ssavas)                                           | [Paper](https://ieeexplore.ieee.org/abstract/document/7987504)                                               |
| Square Root [Pipelined](https://github.com/Ssavas/sqrt-pipelined) / [Not Pipelined](https://github.com/Ssavas/sqrt-no-pipeline)              | Square Root using Harmonized Parabolic Synthesis                | [`@Ssavas`](https://github.com/Ssavas)                                           | [Paper](http://urn.kb.se/resolve?urn=urn:nbn:se:hh:diva-39322)                                               |
| [Pillars](https://github.com/pku-dasys/pillars) | A Consistent CGRA Design Framework | [`@pku-dasys`](https://github.com/pku-dasys) | [Paper](https://woset-workshop.github.io/PDFs/2020/a22.pdf), [Video](https://www.youtube.com/watch?v=oNLyD6koB2g) |
| [Tensil](https://github.com/tensil-ai/tensil) | Machine Learning Accelerators | [`@tensil-ai`](https://github.com/tensil-ai) | [Website](https://www.tensil.ai) |
| [Twine](https://github.com/Twine-Umich/Twine) | A Chisel Extension for Component-Level Heterogeneous Design | [`@shibo-chen`](https://github.com/shibo-chen) | [Paper](https://drive.google.com/file/d/10tDsvv2CSC70GNb_KCFDwXHq4UHdTumm/view) |


### FIRRTL

| Project                                                                                                      | Description                              | Author                                           | Links                                                                                                                                                                                                                                                                           |
|--------------------------------------------------------------------------------------------------------------|------------------------------------------|--------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [MIDAS/DESSERT/Golden Gate](https://github.com/ucb-bar/midas)                                                | FPGA Accelerated Simulation              | [`@ucb-bar`](https://github.com/ucb-bar)         | Papers [1](https://people.eecs.berkeley.edu/~biancolin/papers/carrv17.pdf), [2](http://people.eecs.berkeley.edu/~biancolin/papers/dessert-fpl18.pdf), [3](https://davidbiancolin.github.io/papers/goldengate-iccad19.pdf), [Video](https://www.youtube.com/watch?v=Tvcd4u4_ELM) |
| [Chiffre](https://github.com/IBM/chiffre)                                                                    | Run-time Fault Injection                 | [`@IBM`](https://github.com/IBM)                 | [Paper](https://carrv.github.io/2018/papers/CARRV_2018_paper_2.pdf)                                                                                                                                                                                                             |
| [SIRRTL](https://github.com/apl-cornell/sirrtl)                                                              | Security-typed FIRRTL                    | [`@apl-cornell`](https://github.com/apl-cornell) | [Paper](https://ecommons.cornell.edu/xmlui/bitstream/handle/1813/57673/paper.pdf)                                                                                                                                                                                               |
| [obfuscation](https://github.com/jpsety/obfuscation)                                                         | Transforms to Obfuscate FIRRTL Circuits  | [`@jpsety`](https://github.com/jpsety)           |                                                                                                                                                                                                                                                                                 |
| [Area/Timing Estimates](https://github.com/intel/rapid-design-methods-for-developing-hardware-accelerators/) | Transforms for Area and Timing Estimates | [`@intel`](https://github.com/intel)             | [Video](https://www.youtube.com/watch?v=FktjrjRVBoY)                                                                                                                                                                                                                            |

## Chisel Developers Community

If you want to get *more involved* with the Chisel/FIRRTL ecosystem of projects, feel free to reach out to us on any of the mediums above. If you prefer to dive right in (or have bugs to report), a complete list of the associated Chisel/FIRRTL ecosystem of projects is below:

- [Chisel](https://github.com/freechipsproject/chisel3)
- [FIRRTL](https://github.com/freechipsproject/firrtl)
- [Chisel Testers/`chisel3.iotesters`](https://github.com/freechipsproject/chisel-testers)
- [Chisel Testers2/`chisel3.testers`](https://github.com/ucb-bar/chisel-testers)
- [Treadle](https://github.com/freechipsproject/treadle)
- [Diagrammer](https://github.com/freechipsproject/diagrammer)

## Contributors

Chisel, FIRRTL, and all related projects would not be possible without the contributions of our fantastic developer community.
The following people have contributed to the current release of the projects:

{% include_relative contributors.md %}

## Papers

While Chisel has come a long way since 2012, the original Chisel paper provides some background on motivations and an overview of the (now deprecated) Chisel 2 language:

- [Bachrach, Jonathan, et al. "Chisel: constructing hardware in a scala embedded language." DAC Design Automation Conference 2012. IEEE, 2012.](https://people.eecs.berkeley.edu/~jrb/papers/chisel-dac-2012-corrected.pdf)

The FIRRTL IR and FIRRTL compiler, introduced as part of Chisel 3, are discussed in both the following paper and specification[^historical-caveat]:

- [Izraelevitz, Adam, et al. "Reusability is FIRRTL ground: Hardware construction languages, compiler frameworks, and transformations." Proceedings of the 36th International Conference on Computer-Aided Design. IEEE Press, 2017.](https://aspire.eecs.berkeley.edu/wp/wp-content/uploads/2017/11/Reusability-is-FIRRTL-Ground-Izraelevitz.pdf)
- [Li, Patrick S., Adam M. Izraelevitz, and Jonathan Bachrach. "Specification for the FIRRTL Language." EECS Department, University of California, Berkeley, Tech. Rep. UCB/EECS-2016-9 (2016).](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-9.pdf)

[^historical-caveat]: This specification is provided for historical perspective. For the latest version of the FIRRTL specification you can use [this link](https://github.com/freechipsproject/firrtl/raw/master/spec/spec.pdf).

Finally, Chisel's functional programming and bit-width inference ideas were inspired by earlier work on a hardware description language called *Gel*:

- [Bachrach, Jonathan, Dany Qumsiyeh, and Mark Tobenkin. "Hardware scripting in gel." 2008 16th International Symposium on Field-Programmable Custom Computing Machines. IEEE, 2008.](http://people.eecs.berkeley.edu/~jrb/papers/gel-fccm-2008.pdf)

## Attribution

If you use Chisel in your research, consider citing:

```bib
@inproceedings{bachrach:2012:chisel,
  author={J. {Bachrach} and H. {Vo} and B. {Richards} and Y. {Lee} and A. {Waterman} and R {Avižienis} and J. {Wawrzynek} and K. {Asanović}},
  booktitle={DAC Design Automation Conference 2012},
  title={Chisel: Constructing hardware in a Scala embedded language},
  year={2012},
  volume={},
  number={},
  pages={1212-1221},
  keywords={application specific integrated circuits;C++ language;field programmable gate arrays;hardware description languages;Chisel;Scala embedded language;hardware construction language;hardware design abstraction;functional programming;type inference;high-speed C++-based cycle-accurate software simulator;low-level Verilog;FPGA;standard ASIC flow;Hardware;Hardware design languages;Generators;Registers;Wires;Vectors;Finite impulse response filter;CAD},
  doi={10.1145/2228360.2228584},
  ISSN={0738-100X},
  month={June},}
```

If you use FIRRTL in your research consider citing:

```bib
@INPROCEEDINGS{8203780,
  author={A. Izraelevitz and J. Koenig and P. Li and R. Lin and A. Wang and A. Magyar and D. Kim and C. Schmidt and C. Markley and J. Lawson and J. Bachrach},
  booktitle={2017 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
  title={Reusability is FIRRTL ground: Hardware construction languages, compiler frameworks,
  and transformations},
  year={2017},
  volume={},
  number={},
  pages={209-216},
  keywords={field programmable gate arrays;hardware description languages;program compilers;software reusability;hardware development practices;hardware libraries;open-source hardware intermediate representation;hardware compiler transformations;Hardware construction languages;retargetable compilers;software development;virtual Cambrian explosion;hardware compiler frameworks;parameterized libraries;FIRRTL;FPGA mappings;Chisel;Flexible Intermediate Representation for RTL;Reusability;Hardware;Libraries;Hardware design languages;Field programmable gate arrays;Tools;Open source software;RTL;Design;FPGA;ASIC;Hardware;Modeling;Reusability;Hardware Design Language;Hardware Construction Language;Intermediate Representation;Compiler;Transformations;Chisel;FIRRTL},
  doi={10.1109/ICCAD.2017.8203780},
  ISSN={1558-2434},
  month={Nov},}
```
```bib
@techreport{Li:EECS-2016-9,
    Author = {Li, Patrick S. and Izraelevitz, Adam M. and Bachrach, Jonathan},
    Title = {Specification for the FIRRTL Language},
    Institution = {EECS Department, University of California, Berkeley},
    Year = {2016},
    Month = {Feb},
    URL = {http://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-9.html},
    Number = {UCB/EECS-2016-9}
}
```
