// RUN:  circt-opt -hw-generator-callout='schema-name=Schema_Name generator-executable=node/duh-mem generator-executable-arguments=file1.v,file2.v,file3.v,file4.v' %s

// expected-error @+1 {{cannot find executable 'duh-mem' in path 'node/'}}
module attributes {firrtl.mainModule = "top_mod"}  {
}
