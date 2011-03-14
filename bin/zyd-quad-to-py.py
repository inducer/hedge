#! /usr/bin/env python

from __future__ import with_statement

import sys
with open(sys.argv[1]) as inf:
    lines = [l.strip() for l in inf.readlines() if l.strip()]

rule_name = sys.argv[2]

table = {}

i = 0
while i < len(lines):
    l = lines[i]
    i += 1
    order, point_count = [int(x) for x in l.split()]

    points = []
    weights = []

    for j in xrange(point_count):
        l = lines[i]
        i += 1
        data = [float(x) for x in l.split()]
        points.append(data[:-1])
        weights.append(data[-1])

    table[order] = { 
            "points": points,
            "weights": weights }

from pprint import pformat
print "%s = %s" % (rule_name, pformat(table))

print """

%s = dict(
    (order, dict((name, numpy.array(ary)) for name, ary in rule.iteritems()))
    for order, rule in %s.iteritems())
""" % (rule_name, rule_name)
