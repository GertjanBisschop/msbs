import random
import math
import dataclasses
from typing import List
from typing import Any

import numpy as np
import tskit


@dataclasses.dataclass
class AncestryInterval:
    """
    Records that the specified interval contains genetic material ancestral
    to the specified number of samples.
    """

    left: int
    right: int
    ancestral_to: int
    value: int = 0

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
    value: float = 1.0

    def __str__(self):
        s = f"{self.node}:["
        for interval in self.ancestry:
            s += str(
                (interval.left, interval.right, interval.ancestral_to, interval.value)
            )
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

    def set_b(self, b_map):
        b = 0.0
        cumspan = 0.0
        m = len(self.ancestry)
        n = len(b_map.rate)
        i = 0  # interval index
        j = 1  # b_map index
        left = 0

        while i < m:
            left = max(self.ancestry[i].left, left)
            if left >= b_map.position[j]:
                if j < n:
                    j += 1
            else:
                right = min(b_map.position[j], self.ancestry[i].right)
                span = right - left
                b += b_map.rate[j - 1] * span
                cumspan += span
                if right == self.ancestry[i].right:
                    i += 1
                if right == b_map.position[j]:
                    j += 1
                left = right

        self.value = b / cumspan

    def split(self, breakpoint, b_map):
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
        self.set_b(b_map)
        right_lin = Lineage(self.node, right_ancestry)

        return right_lin


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


def pairwise_products(v: np.ndarray):
    assert len(v.shape) == 1
    n = v.shape[0]
    m = v.reshape(n, 1) @ v.reshape(1, n)

    return m[np.tril_indices_from(m, k=-1)].ravel()


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
            ret += (self.position[i + 1] - self.position[i]) * self.rate[i]
            i += 1
        return ret


@dataclasses.dataclass
class BMap(RateMap):
    pass


@dataclasses.dataclass
class SuperSimulator:
    L: float
    r: float
    n: int
    Ne: float
    ploidy: int = 2
    seed: int = None

    def stop_condition(self):
        n = self.n * self.ploidy
        for lineage in self.lineages:
            for segment in lineage.ancestry:
                if segment.ancestral_to < n:
                    return False
        return True

    def finalise(self, tables, nodes, simplify):
        for node in nodes:
            tables.nodes.add_row(
                flags=node.flags, time=node.time, metadata=node.metadata
            )
        tables.sort()
        tables.edges.squash()
        ts = tables.tree_sequence()
        if simplify:
            ts = ts.simplify()

        return ts


@dataclasses.dataclass
class Simulator(SuperSimulator):
    B: BMap = None
    s: float = None
    model: str = "localne"

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        if self.B is None:
            position = np.zeros(2)
            position[-1] = self.L
            self.B = BMap(position, np.ones(1))
        self.lineages = []
        self.num_lineages = 0

    def common_ancestor_waiting_time_from_rate(self, rate):
        u = self.rng.expovariate(rate)
        # incorporate info from B-map
        return self.ploidy * self.Ne * u

    def common_ancestor_waiting_time(self):
        # perform all pairwise weighted contributions
        n = self.num_lineages
        rate = np.sum(pairwise_products(np.array(self.coal_rates)))
        return self.common_ancestor_waiting_time_from_rate(rate)

    def remove_lineage(self, lineage_id):
        lin = self.lineages.pop(lineage_id)
        if self.model == "localne":
            _ = self.coal_rates.pop(lineage_id)
        self.num_lineages -= 1
        return lin

    def insert_lineage(self, lineage):
        if self.model == "localne":
            lineage.set_b(self.B)
        self.lineages.append(lineage)
        self.num_lineages += 1

    def stop_condition(self):
        """
        Returns True if all segments are ancestral to n samples in all
        lineages.
        """
        if self.model == "zeng":
            return False
        else:
            return super().stop_condition()

    def run(self, simplify=True):
        if self.model == "localne":
            return self._sim_local_ne(simplify)
        elif self.model == "zeng":
            assert self.s is not None
            return self._sim_zeng(simplify)
        else:
            raise ValueError("Model not implemented.")

    def _sim_local_ne(self, simplify=True):
        """
        Experimental implementation of coalescent with local Ne map along genome.

        NOTE! This hasn't been statistically tested and is probably not correct.
        """

        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        nodes = []
        for _ in range(self.n * self.ploidy):
            segment_chain = [AncestryInterval(0, self.L, 1)]
            b_value = self.B.weighted_average()
            self.insert_lineage(Lineage(len(nodes), segment_chain, b_value))
            nodes.append(Node(time=0, flags=tskit.NODE_IS_SAMPLE))

        t = 0
        while not self.stop_condition():
            lineage_links = [
                lineage.num_recombination_links for lineage in self.lineages
            ]
            total_links = sum(lineage_links)
            re_rate = total_links * self.r
            t_re = math.inf if re_rate == 0 else self.rng.expovariate(re_rate)
            self.coal_rates = [1 / lineage.value for lineage in self.lineages]
            t_ca = self.common_ancestor_waiting_time()
            t_inc = min(t_re, t_ca)
            t += t_inc

            if t_inc == t_re:  # recombination
                left_lineage = self.rng.choices(self.lineages, weights=lineage_links)[0]
                breakpoint = self.rng.randrange(
                    left_lineage.left + 1, left_lineage.right
                )
                assert left_lineage.left < breakpoint < left_lineage.right
                right_lineage = left_lineage.split(breakpoint, self.B)
                self.insert_lineage(right_lineage)
                child = left_lineage.node
                assert right_lineage.node == child

            else:  # common ancestor event
                a = self.remove_lineage(
                    self.rng.choices(range(self.num_lineages), weights=self.coal_rates)[
                        0
                    ]
                )
                b = self.remove_lineage(
                    self.rng.choices(range(self.num_lineages), weights=self.coal_rates)[
                        0
                    ]
                )
                c = Lineage(len(nodes), [])
                for interval, intersecting_lineages in merge_ancestry([a, b]):
                    # if interval.ancestral_to < n:
                    c.ancestry.append(interval)
                    for lineage in intersecting_lineages:
                        tables.edges.add_row(
                            interval.left, interval.right, c.node, lineage.node
                        )

                nodes.append(Node(time=t))
                self.insert_lineage(c)

        return self.finalise(tables, nodes, simplify)

    def _sim_zeng(self, simplify=True):
        """
        Simulate under the model described by Zeng and Charlesworth 2011

        NOTE! This hasn't been statistically tested and is probably not correct.
        """
        rng = np.random.default_rng(self.rng.randint(1, 2**16))
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        nodes = []
        for _ in range(self.n * self.ploidy):
            num_mutations = rng.poisson()
            segment_chain = [AncestryInterval(0, self.L, 1, num_mutations)]
            self.insert_lineage(Lineage(len(nodes), segment_chain))
            nodes.append(Node(time=0, flags=tskit.NODE_IS_SAMPLE))

        t = 0
        while not self.fully_coalesced():
            lineage_links = [
                lineage.num_recombination_links for lineage in self.lineages
            ]
            total_links = sum(lineage_links)
            re_rate = total_links * self.r
            t_re = math.inf if re_rate == 0 else self.rng.expovariate(re_rate)
            t_ca = self.common_ancestor_waiting_time()
            t_inc = min(t_re, t_ca)
            t += t_inc

            if t_inc == t_re:  # recombination
                left_lineage = self.rng.choices(self.lineages, weights=lineage_links)[0]
                breakpoint = self.rng.randrange(
                    left_lineage.left + 1, left_lineage.right
                )
                assert left_lineage.left < breakpoint < left_lineage.right
                right_lineage = left_lineage.split(breakpoint, self.B)
                self.insert_lineage(right_lineage)
                child = left_lineage.node
                assert right_lineage.node == child

            else:  # common ancestor event
                a = self.remove_lineage(self.rng.randrange(self.num_lineages))
                b = self.remove_lineage(self.rng.randrange(self.num_lineages))
                c = Lineage(len(nodes), [])
                for interval, intersecting_lineages in merge_ancestry([a, b]):
                    # if interval.ancestral_to < n:
                    c.ancestry.append(interval)
                    for lineage in intersecting_lineages:
                        tables.edges.add_row(
                            interval.left, interval.right, c.node, lineage.node
                        )

                nodes.append(Node(time=t))
                self.insert_lineage(c)

        return self.finalise(tables, nodes, simplify)
