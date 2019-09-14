#!/usr/bin/env python3
# See LICENSE for license details.

import subprocess
import re
from statistics import median, stdev
import sys
import argparse
from collections import OrderedDict
import os
import numbers

# Currently hardcoded
def get_firrtl_repo():
    cmd = ['git', 'rev-parse', '--show-toplevel']
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    assert result.returncode == 0
    return result.stdout.rstrip()

firrtl_repo = get_firrtl_repo()

platform = ""
if sys.platform == 'darwin':
    print("Running on MacOS")
    platform = 'macos'
elif sys.platform.startswith("linux"):
    print("Running on Linux")
    platform = 'linux'
else :
    raise Exception('Unrecognized platform ' + sys.platform)

def time():
    if platform == 'macos':
        return ['/usr/bin/time', '-l']
    if platform == 'linux':
        return ['/usr/bin/time', '-v']

def extract_max_size(output):
    regex = ''
    if platform == 'macos':
        regex = '(\d+)\s+maximum resident set size'
    if platform == 'linux':
        regex = 'Maximum resident set size[^:]*:\s+(\d+)'

    m = re.search(regex, output, re.MULTILINE)
    if m :
        return int(m.group(1))
    else :
        raise Exception('Max set size not found!')

def extract_run_time(output):
    regex = ''
    res = None
    if platform == 'macos':
        regex = '(\d+\.\d+)\s+real'
    if platform == 'linux':
        regex = 'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'
    m = re.search(regex, output, re.MULTILINE)
    if m :
        text = m.group(1)
        if platform == 'macos':
            return float(text)
        if platform == 'linux':
            parts = text.split(':')
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[0])
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
    raise Exception('Runtime not found!')

def run_firrtl(java, jar, design):
    java_cmd = java.split()
    cmd = time() + java_cmd + ['-cp', jar, 'firrtl.stage.FirrtlMain', '-i', design,'-o','out.v','-X','verilog']
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    if result.returncode != 0 :
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    size = extract_max_size(result.stderr.decode('utf-8'))
    runtime = extract_run_time(result.stderr.decode('utf-8'))
    return (size, runtime)

def parseargs():
    parser = argparse.ArgumentParser("Benchmark FIRRTL")
    parser.add_argument('--designs', type=str, nargs='+',
                        help='FIRRTL input files to use as benchmarks')
    parser.add_argument('--versions', type=str, nargs='+', default=['HEAD'],
                        help='FIRRTL commit hashes to benchmark')
    parser.add_argument('--iterations', '-N', type=int, default=10,
                        help='Number of times to run each benchmark')
    parser.add_argument('--jvms', type=str, nargs='+', default=['java'],
                        help='JVMs to use')
    return parser.parse_args()

def get_version_hashes(versions):
    res = subprocess.run(['git', '-C', firrtl_repo, 'fetch'])
    assert res.returncode == 0, '"{}" must be an existing repo!'.format(firrtl_repo)
    hashes = OrderedDict()
    for version in versions :
        res = subprocess.run(['git', '-C', firrtl_repo, 'rev-parse', '--short', version], stdout=subprocess.PIPE)
        assert res.returncode == 0, '"{}" is not a legal revision!'.format(version)
        hashcode = res.stdout.decode('utf-8').rstrip()
        if hashcode in hashes :
            print('{} and {} are the same revision!'.format(version, hashes[hashcode]))
        else :
            hashes[hashcode] = version
    return hashes

def clone_and_build(hashcode):
    repo = 'firrtl.{}'.format(hashcode)
    jar = 'firrtl.{}.jar'.format(hashcode)
    if os.path.exists(jar):
        print('{} already exists, skipping jar creation'.format(jar))
    else :
        if os.path.exists(repo):
            assert os.path.isdir(repo), '{} already exists but isn\'t a directory!'.format(repo)
        else :
            res = subprocess.run(['git', 'clone', firrtl_repo, repo])
            assert res.returncode == 0
        res = subprocess.run(['git', '-C', repo, 'checkout', hashcode])
        assert res.returncode == 0
        res = subprocess.run(['make', '-C', repo, 'build-scala'])
        assert res.returncode == 0
        res = subprocess.run(['cp', '{}/utils/bin/firrtl.jar'.format(repo), jar])
        assert res.returncode == 0
        res = subprocess.run(['rm', '-rf', repo])
        assert res.returncode == 0
    return jar

def build_firrtl_jars(versions):
    jars = OrderedDict()
    for hashcode, version in versions.items():
        jars[hashcode] = clone_and_build(hashcode)
    return jars

def check_designs(designs):
    for design in designs:
        assert os.path.exists(design), '{} must be an existing file!'.format(design)

# /usr/bin/time -v on Linux returns size in kbytes
# /usr/bin/time -l on MacOS returns size in Bytes
def norm_max_set_sizes(sizes):
    div = None
    if platform == 'linux':
        d = 1000.0
    if platform == 'macos':
        d = 1000000.0
    return [s / d for s in sizes]

def main():
    args = parseargs()
    designs = args.designs
    check_designs(designs)
    hashes = get_version_hashes(args.versions)
    jars = build_firrtl_jars(hashes)
    jvms = args.jvms
    N = args.iterations
    info = [['java', 'revision', 'design', 'max heap', 'SD', 'runtime', 'SD']]
    for java in jvms:
        print("Running with '{}'".format(java))
        for hashcode, jar in jars.items():
            print("Benchmarking {}...".format(hashcode))
            revision = hashcode
            java_title = java
            for design in designs:
                print('Running {}...'.format(design))
                (sizes, runtimes) = zip(*[run_firrtl(java, jar, design) for i in range(N)])
                norm_sizes = norm_max_set_sizes(sizes)
                info.append([java_title, revision, design, median(norm_sizes), stdev(norm_sizes), median(runtimes), stdev(runtimes)])
                java_title = ''
                revision = ''

    for line in info:
        formated = ['{:0.2f}'.format(elt) if isinstance(elt, numbers.Real) else elt for elt in line]
        print(','.join(formated))

if __name__ == '__main__':
    main()
