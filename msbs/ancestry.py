from __future__ import annotations

import random
import math
import dataclasses
from typing import List
from typing import Any

import numpy as np
import tskit

NODE_IS_RECOMB = 1 << 1


# AncestryInterval is the equivalent of msprime's Segment class. The
# important different here is that we don't associated nodes with
# individual intervals here: because this is an ARG, nodes that
# we pass through are recorded.
#
# (The ancestral_to field is also different here, but that's because
# I realised that the way we're tracking extant ancestral material
# in msprime is unnecessarily complicated, and we can actually
# track it locally. There is potentially quite a large performance
# increase available in msprime from this.)


@dataclasses.dataclass
class AncestryInterval:
    """
    Records that the specified interval contains genetic material ancestral
    to the specified number of samples.
    """

    left: int
    right: int
    ancestral_to: int

    @property
    def span(self):
        return self.right - self.left


@dataclasses.dataclass
class Lineage:
    """
    A single lineage that is present during the simulation of the coalescent
    with recombination. The node field represents the last (as we go backwards
    in time) genome in which an ARG event occured. That is, we can imagine
    a lineage representing the passage of the ancestral material through
    a sequence of ancestral genomes in which it is not modified.
    """

    node: int
    ancestry: List[AncestryInterval]
    b: float = 1.0

    def __str__(self):
        s = f"{self.node}:["
        for interval in self.ancestry:
            s += str((interval.left, interval.right, interval.ancestral_to))
            s += ", "
        if len(self.ancestry) > 0:
            s = s[:-2]
        return s + "]"

    @property
    def num_recombination_links(self):
        """
        The number of positions along this lineage's genome at which a recombination
        event can occur.
        """
        return self.right - self.left - 1

    @property
    def left(self):
        """
        Returns the leftmost position of ancestral material.
        """
        return self.ancestry[0].left

    @property
    def right(self):
        """
        Returns the rightmost position of ancestral material.
        """
        return self.ancestry[-1].right

    def split(self, breakpoint):
        """
        Splits the ancestral material for this lineage at the specified
        breakpoint, and returns a second lineage with the ancestral
        material to the right.
        """
        left_ancestry = []
        right_ancestry = []
        for interval in self.ancestry:
            if interval.right <= breakpoint:
                left_ancestry.append(interval)
            elif interval.left >= breakpoint:
                right_ancestry.append(interval)
            else:
                assert interval.left < breakpoint < interval.right
                left_ancestry.append(dataclasses.replace(interval, right=breakpoint))
                right_ancestry.append(dataclasses.replace(interval, left=breakpoint))
        self.ancestry = left_ancestry
        return Lineage(self.node, right_ancestry)


# The details of the machinery in the next two functions aren't important.
# It could be done more cleanly and efficiently. The basic idea is that
# we're providing a simple way to find the overlaps in the ancestral
# material of two or more lineages, abstracting the complex interval
# logic out of the main simulation.
@dataclasses.dataclass
class MappingSegment:
    left: int
    right: int
    value: Any = None


def overlapping_segments(segments):
    """
    Returns an iterator over the (left, right, X) tuples describing the
    distinct overlapping segments in the specified set.
    """
    S = sorted(segments, key=lambda x: x.left)
    n = len(S)
    # Insert a sentinel at the end for convenience.
    S.append(MappingSegment(math.inf, 0))
    right = S[0].left
    X = []
    j = 0
    while j < n:
        # Remove any elements of X with right <= left
        left = right
        X = [x for x in X if x.right > left]
        if len(X) == 0:
            left = S[j].left
        while j < n and S[j].left == left:
            X.append(S[j])
            j += 1
        j -= 1
        right = min(x.right for x in X)
        right = min(right, S[j + 1].left)
        yield left, right, X
        j += 1

    while len(X) > 0:
        left = right
        X = [x for x in X if x.right > left]
        if len(X) > 0:
            right = min(x.right for x in X)
            yield left, right, X


def merge_ancestry(lineages):
    """
    Return an iterator over the ancestral material for the specified lineages.
    For each distinct interval at which ancestral material exists, we return
    the AncestryInterval and the corresponding list of lineages.
    """
    # See note above on the implementation - this could be done more cleanly.
    segments = []
    for lineage in lineages:
        for interval in lineage.ancestry:
            segments.append(
                MappingSegment(interval.left, interval.right, (lineage, interval))
            )

    for left, right, U in overlapping_segments(segments):
        ancestral_to = sum(u.value[1].ancestral_to for u in U)
        interval = AncestryInterval(left, right, ancestral_to)
        yield interval, [u.value[0] for u in U]


@dataclasses.dataclass
class Node:
    time: float
    flags: int = 0
    metadata: dict = dataclasses.field(default_factory=dict)


def fully_coalesced(lineages, n):
    """
    Returns True if all segments are ancestral to n samples in all
    lineages.
    """
    for lineage in lineages:
        for segment in lineage.ancestry:
            if segment.ancestral_to < n:
                return False
    return True

@dataclasses.dataclass
class RateMap:
    position: np.ndarray
    rate: np.ndarray

    """
    uniform RateMap(position=[0, sequence_length], rate=[rate])
    """

    def weighted_average(self):
        ret = 0
        i = 0
        while i < self.rate.size:
            ret += (self.position[i+1] - self.position[i]) * self.rate[i]
            i += 1
        return ret

@dataclasses.dataclass
class BMap(RateMap):
    
    def init_b_value(self):
        return self.weighted_average()

@dataclasses.dataclass
class Simulator:
    L: float
    rho: float
    n: int
    Ne: float
    ploidy: int = 2
    seed: int = None
    B: BMap = None

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        if self.B is None:
            position = np.zeros(2)
            position[-1] = self.L
            self.B = BMap(position, np.ones(1))

    def common_ancestor_waiting_time_from_rate(self, rate):
        u = self.rng.expovariate(rate)
        return self.ploidy * self.Ne * u
    
    def common_ancestor_waiting_time(self, lineages):
        n = len(lineages)
        rate = n * (n - 1) / 2
        return self.common_ancestor_waiting_time_from_rate(rate)

    def sim_coalescent(self):
        """
        Simulate under the coalescent with recombination
        and return the tskit TreeSequence object.

        NOTE! This hasn't been statistically tested and is probably not correct.
        """
        
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        lineages = []
        nodes = []
        for _ in range(self.n * self.ploidy):
            segment_chain = [AncestryInterval(0, self.L, 1)]
            b_value = self.B.init_b_value()
            lineages.append(Lineage(len(nodes), segment_chain, b_value))
            nodes.append(Node(time=0, flags=tskit.NODE_IS_SAMPLE))

        t = 0
        while not fully_coalesced(lineages, self.n * self.ploidy):
            lineage_links = [lineage.num_recombination_links for lineage in lineages]
            total_links = sum(lineage_links)
            re_rate = total_links * self.rho
            t_re = math.inf if re_rate == 0 else self.rng.expovariate(re_rate)
            t_ca = self.common_ancestor_waiting_time(lineages)
            t_inc = min(t_re, t_ca)
            t += t_inc

            if t_inc == t_re: # recombination
                left_lineage = self.rng.choices(lineages, weights=lineage_links)[0]
                breakpoint = self.rng.randrange(left_lineage.left + 1, left_lineage.right)
                assert left_lineage.left < breakpoint < left_lineage.right
                right_lineage = left_lineage.split(breakpoint)
                lineages.append(right_lineage)
                child = left_lineage.node
                assert right_lineage.node == child

            else: # common ancestor event
                a = lineages.pop(self.rng.randrange(len(lineages)))
                b = lineages.pop(self.rng.randrange(len(lineages)))
                c = Lineage(len(nodes), [])
                for interval, intersecting_lineages in merge_ancestry([a, b]):
                    # if interval.ancestral_to < n:
                    c.ancestry.append(interval)
                    for lineage in intersecting_lineages:
                        tables.edges.add_row(
                            interval.left, interval.right, c.node, lineage.node
                        )

                nodes.append(Node(time=t))
                # if len(c.ancestry) > 0:
                lineages.append(c)

        for node in nodes:
            tables.nodes.add_row(flags=node.flags, time=node.time, metadata=node.metadata)
        tables.sort()
        # TODO not sure if this is the right thing to do, but it makes it easier
        # to compare with examples.
        tables.edges.squash()
        return tables.tree_sequence()


@dataclasses.dataclass
class Individual:
    id: int = -1
    lineages: List[Lineage] = dataclasses.field(default_factory=list)
    collected_lineages: List[List[Lineage]] = dataclasses.field(
        default_factory=lambda: [[], []]
    )


class IntervalSet:
    """
    Naive and simple implementation of discrete intervals.
    """

    def __init__(self, L, tuples=None):
        assert int(L) == L
        self.I = np.zeros(int(L), dtype=int)
        if tuples is not None:
            for left, right in tuples:
                self.insert(left, right)

    def __str__(self):
        return str(self.I)

    def __repr__(self):
        return repr(list(self.I))

    def __eq__(self, other):
        return np.array_equal(self.I == 0, other.I == 0)

    def insert(self, left, right):
        assert int(left) == left
        assert int(right) == right
        self.I[int(left) : int(right)] = 1

    def contains(self, x):
        assert int(x) == x
        return self.I[int(x)] != 0

    def union(self, other):
        """
        Returns a new IntervalSet with the union of intervals in this and
        other.
        """
        new = IntervalSet(self.I.shape[0])
        assert other.I.shape == self.I.shape
        new.I[:] = np.logical_or(self.I, other.I)
        return new

    def intersection(self, other):
        """
        Returns a new IntervalSet with the intersection of intervals in this and
        other.
        """
        new = IntervalSet(self.I.shape[0])
        assert other.I.shape == self.I.shape
        new.I[:] = np.logical_and(self.I, other.I)
        return new

    def is_subset(self, other):
        """
        Return True if this set is a subset of other.
        """
        a = np.all(other.I[self.I == 1] == 1)
        b = np.all(self.I[other.I == 0] == 0)
        return a and b


@dataclasses.dataclass
class RecombinationEvent:
    parent_edges: List[tskit.Edge] = dataclasses.field(default_factory=list)
    child_edge: tskit.Edge | None = None
