#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

import sys
import os
import re
from itertools import *
from collections import namedtuple
import argparse

Transform = namedtuple('Transform', ['name', 'children', 'runtime'])

start_xform_re = re.compile(r'\s*=+\s+Starting\s+Transform\s+(\S+)\s+=+\s*')
finish_xform_re = re.compile(r'\s*=+\s+Finished\s+Transform\s+(\S+)\s+=+\s*')
time_re = re.compile(r'\s*Time:\s*(\d+(\.\d+)?)\s*ms\s*')


def get_start(line):
    if line is not None:
        m = start_xform_re.match(line)
        if m:
            return m.group(1)
    return None


def get_finish(line):
    m = finish_xform_re.match(line)
    if m:
        return m.group(1)
    return None


def get_time(line):
    m = time_re.match(line)
    if m:
        return m.group(1)
    return None


def safe_next(it):
    try:
        return next(it)
    except StopIteration:
        return None


def non_empty(it):
    return any(True for _ in it)


def read_transform(name, lines):
    # The log may be malformed so we just catch that whenever
    try:
        children = []
        time = None
        while True:
            line = next(lines)
            start_opt = get_start(line)
            if start_opt is not None:
                child = read_transform(start_opt, lines)
                children.append(child)
            else:
                time_opt = get_time(line)
                if time_opt is not None:
                    time = time_opt
                else:
                    end_opt = get_finish(line)
                    if end_opt is not None:
                        assert end_opt == name
                        return Transform(name, children, time)
    except StopIteration:
        return None


def read_top_transform(it):
    # Find start
    lines = dropwhile(lambda line: not start_xform_re.match(line), it)
    name = get_start(safe_next(lines))
    if name is not None:
        return read_transform(name, lines)
    else:
        return None


def pretty_transform(xform):
    time_str = " ({} ms)".format(xform.runtime) if xform.runtime is not None else ""
    lines = [xform.name + time_str]
    nchildren = len(xform.children)
    for i in range(0, nchildren):
        child = xform.children[i]
        child_lines = pretty_transform(child)
        if i == nchildren - 1:
            branch = "└── "
            indent = "    "
        else:
            branch = "├── "
            indent = "│   "
        first = True
        for line in child_lines:
            prefix = branch if first else indent
            lines.append(prefix + line)
            first = False
    return lines


def remove_time(xform):
    children = [remove_time(child) for child in xform.children]
    return Transform(xform.name, children, None)


def is_existing_file(name):
    if not os.path.isfile(name):
        msg = "File {} does not exist!".format(name)
        raise argparse.ArgumentTypeError(msg)
    else:
        return name


def get_parser():
    parser = argparse.ArgumentParser(description='Parse FIRRTL transform log')
    parser.add_argument('file', help='log file',
                        type=is_existing_file)
    parser.add_argument('-s', '--strip-time', action='store_true', default=False,
                         help='Strip time values')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    transforms = []
    with open(args.file, "r") as f:
        while True:
            xform = read_top_transform(f)
            if xform is None:
                break
            else:
                transforms.append(xform)
    if len(transforms) == 0:
        print("{} does not contain a FIRRTL Transform log!".format(args.file))
        sys.exit(-1)
    top = Transform("firrtl.stage.transforms.Compiler", transforms, None)
    if args.strip_time:
        top = remove_time(top)
    pretty = pretty_transform(top)
    for line in pretty:
        print(line)

