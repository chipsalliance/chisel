# firrtl
#### Flexible Internal Representation for RTL

 This repository contains the compiler for .fir files.
 It is currently being developed in stanza, with the end-goal to translate into scala for ease of other people to use.
 This decision was made for multiple reasons:  
       1.   Previous development had already been done in stanza   
       2.   Most of the passes are relatively language independent   
       3.   Stanza is easier to develop in than scala bc less-strict type system   
       4.   As a favor, its useful to develop in stanza to give Patrick some language tips :)   
 The hardest part to port will likely be the parser, but we hope to use an existing scala-based general-purpose parser for the port.   

#### Installation instructions (for Mac):    
*Disclaimer*: This project is going through development stages so there is no guarantee anything works.     
    `git clone https://github.com/ucb-bar/firrtl # Clone repository`     
    `make install-mac # Stanza installation`     
    `pip install lit # Install lit (this assumes you have pip installed)`     
    `make build # Build firrtl`     
    `make check # Run tests`     
    `make # Build & test`     

#### Scala implementation 
The Scala FIRRTL implementation relies upon sbt 0.13.6. It uses sbt-assembly to create a fat JAR.    
Using a bash script and a symbolic link it can be used with the same command-line arguments as the stanza implementation.    
Example use:    
  `make build-scala`    
  `make set-scala` # Creates symbolic link, make set-stanza reverts to stanza implementation       
  `./utils/bin/firrtl -i <input> -o <output> -X <compiler>`    
