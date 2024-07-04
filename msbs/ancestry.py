import random
import math
import itertools
import dataclasses
import numpy as np
import tskit

from typing import List, Any
from msbs import utils


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
        s = f"{self.node}: {self.value} ["
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

    def set_value(self, map):
        self.value = map.get(self)

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
        right_lin = Lineage(self.node, right_ancestry, self.value)

        return right_lin

    def intersect(self, other):
        """
        Returns list with the overlap between the ancestry intervals
        of Lineages a and b.
        """
        n = len(self.ancestry)
        m = len(other.ancestry)
        i = j = 0
        overlap = []
        overlap_length = 0
        while i < n and j < m:
            if self.ancestry[i].right <= other.ancestry[j].left:
                i += 1
            elif self.ancestry[i].left >= other.ancestry[j].right:
                j += 1
            else:
                left = max(self.ancestry[i].left, other.ancestry[j].left)
                right = min(self.ancestry[i].right, other.ancestry[j].right)
                overlap.append(
                    AncestryInterval(
                        left,
                        right,
                        self.ancestry[i].ancestral_to + other.ancestry[j].ancestral_to,
                    )
                )
                overlap_length += right - left
                if self.ancestry[i].right < other.ancestry[j].right:
                    i += 1
                else:
                    j += 1

        return (overlap, overlap_length)


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


@dataclasses.dataclass
class RateMap:
    position: np.ndarray
    rate: np.ndarray

    """
    uniform RateMap(position=[0, sequence_length], rate=[rate])
    """

    def weighted_average(self, left=0.0, right=None, normalise=True, total=False):
        if right is None:
            right = self.position[-1]

        # Ensure left and right are within the bounds of position
        left = max(left, self.position[0])
        right = min(right, self.position[-1])

        ret = 0.0
        total_length = 0.0
        i = 0

        while i < self.rate.size:
            segment_start = max(left, self.position[i])
            segment_end = min(right, self.position[i + 1])

            if segment_start < segment_end:
                segment_length = segment_end - segment_start
                ret += segment_length * self.rate[i]
                total_length += segment_length

            if self.position[i + 1] >= right:
                break
            i += 1

        if total_length == 0.0:
            return 0.0
        if normalise:
            if total:
                total_length = self.position[-1]
            ret /= total_length

        return ret

    def intersect_lineage(self, ancestry):
        total = 0

        for interval in ancestry:
            i = np.searchsorted(self.position, interval.left, side="right")
            j = np.searchsorted(self.position, interval.right, side="right")

            while i <= j and i < self.position.size:
                left = max(interval.left, self.position[i - 1])
                right = min(interval.right, self.position[i])
                total += self.rate[i - 1] * (right - left)
                i += 1

        return total


@dataclasses.dataclass
class BMap(RateMap):
    def get(self, lineage):
        b = 0.0
        cumspan = 0.0
        m = len(lineage.ancestry)
        n = len(self.rate)
        i = 0  # interval index
        j = 1  # b_map index
        left = 0

        while i < m:
            left = max(lineage.ancestry[i].left, left)
            if left >= self.position[j]:
                if j < n:
                    j += 1
            else:
                right = min(self.position[j], lineage.ancestry[i].right)
                span = right - left
                b += self.rate[j - 1] * span
                cumspan += span
                if right == lineage.ancestry[i].right:
                    i += 1
                if right == self.position[j]:
                    j += 1
                left = right

        return b / cumspan


@dataclasses.dataclass
class FitnessClassMap(RateMap):
    def __post_init__(self):
        self.av = self.weighted_average()

    def get(self, lineage, left=0.0, right=None):
        left = max(left, self.position[0])
        right = min(right, self.position[-1])
        return self.weighted_average(left, right)


@dataclasses.dataclass
class SuperSimulator:
    L: float
    r: float
    n: int
    Ne: float
    ploidy: int = 2
    seed: int = None

    def __post_init__(self):
        pass

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

    def reset(self, seed=None):
        self.seed = seed
        self.__post_init__()


@dataclasses.dataclass
class Simulator(SuperSimulator):
    B: BMap = None
    model: str = "localne"

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        if self.B is None:
            position = np.zeros(2)
            position[-1] = self.L
            self.B = BMap(position, np.ones(1))
        self.lineages = []
        self.num_lineages = 0

    def print_state(self):
        for lineage in self.lineages:
            print(lineage)
        print("------------------------")

    def common_ancestor_waiting_time_from_rate(self, rate):
        u = self.rng.expovariate(rate)
        return self.ploidy * self.Ne * u

    def common_ancestor_waiting_time(self):
        # perform all pairwise weighted contributions
        n = self.num_lineages
        rate = np.sum(utils.pairwise_products(np.array(self.coal_rates)))
        return self.common_ancestor_waiting_time_from_rate(rate)

    def remove_lineage(self, lineage_id):
        lin = self.lineages.pop(lineage_id)
        if self.model == "localne":
            _ = self.coal_rates.pop(lineage_id)
        self.num_lineages -= 1
        return lin

    def insert_lineage(self, lineage):
        if self.model == "localne":
            lineage.set_value(self.B)
        self.lineages.append(lineage)
        self.num_lineages += 1

    def stop_condition(self):
        """
        Returns True if all segments are ancestral to n samples in all
        lineages.
        """
        return super().stop_condition()

    def run(self, simplify=True):
        if self.model == "localne":
            return self._sim_local_ne(simplify)
        elif self.model == "overlap":
            return self._sim_local_ne_overlap(simplify)
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
            self.insert_lineage(Lineage(len(nodes), segment_chain))
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
                right_lineage = left_lineage.split(breakpoint)
                left_lineage.set_value(self.B)
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

    def _sim_local_ne_overlap(self, simplify=True):
        """
        Experimental implementation of coalescent with local Ne map along genome.
        Compute coalescence time based on weighted mean of B-map in overlapping regions
        NOTE! This hasn't been statistically tested and is probably not correct.
        """

        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        nodes = []
        for _ in range(self.n * self.ploidy):
            segment_chain = [AncestryInterval(0, self.L, 1)]
            self.insert_lineage(Lineage(len(nodes), segment_chain))
            nodes.append(Node(time=0, flags=tskit.NODE_IS_SAMPLE))

        t = 0
        while not self.stop_condition():
            lineage_links = [
                lineage.num_recombination_links for lineage in self.lineages
            ]
            total_links = sum(lineage_links)
            re_rate = total_links * self.r
            t_re = math.inf if re_rate == 0 else self.rng.expovariate(re_rate)
            # compute coal rate for each pair
            self.coal_rates = np.zeros(math.comb(self.num_lineages, 2))
            for pair in itertools.combinations(range(self.num_lineages), 2):
                pair_idx = utils.combinadic_map(pair)
                overlap_ancestry, overlap = self.lineages[pair[0]].intersect(
                    self.lineages[pair[1]]
                )
                if overlap > 0:
                    self.coal_rates[pair_idx] = overlap / self.B.intersect_lineage(
                        overlap_ancestry
                    )
            ca_rate = np.sum(self.coal_rates)
            assert ca_rate > 0
            t_ca = self.common_ancestor_waiting_time_from_rate(ca_rate)
            t_inc = min(t_re, t_ca)
            assert t_inc < math.inf
            t += t_inc

            if t_inc == t_re:  # recombination
                left_lineage = self.rng.choices(self.lineages, weights=lineage_links)[0]
                breakpoint = self.rng.randrange(
                    left_lineage.left + 1, left_lineage.right
                )
                assert left_lineage.left < breakpoint < left_lineage.right
                right_lineage = left_lineage.split(breakpoint)
                self.insert_lineage(right_lineage)
                child = left_lineage.node
                assert right_lineage.node == child

            else:  # common ancestor event
                # pick lineages based on coal_rates
                random_idx = self.rng.choices(
                    range(self.coal_rates.size), weights=self.coal_rates
                )[0]
                coal_pair = [
                    self.remove_lineage(lin_idx)
                    for lin_idx in utils.reverse_combinadic_map(random_idx)
                ]
                c = Lineage(len(nodes), [])
                for interval, intersecting_lineages in merge_ancestry(coal_pair):

                    c.ancestry.append(interval)
                    for lineage in intersecting_lineages:
                        tables.edges.add_row(
                            interval.left, interval.right, c.node, lineage.node
                        )

                nodes.append(Node(time=t))
                self.insert_lineage(c)

        return self.finalise(tables, nodes, simplify)
