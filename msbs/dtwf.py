import bintrees
import collections
import dataclasses
import heapq
import math
import numpy as np
import random
import tskit

from msbs import ancestry
from msbs import utils


class Individual:
    """
    Class representing a diploid individual in the DTWF pedigree model.
    """

    def __init__(self, id_, *, ploidy, nodes, parents, time):
        self.id = id_
        self.ploidy = ploidy
        self.nodes = nodes
        self.parents = parents
        self.time = time
        # self.lineages = [ancestry.Lineage(None, None) for i in range(ploidy)]
        self.common_ancestors = [[] for i in range(ploidy)]

    def __str__(self):
        return (
            f"(ID: {self.id}, time: {self.time}, "
            + f"parents: {self.parents}, nodes: {self.nodes}, "
            + f"common_ancestors: {self.common_ancestors})"
        )

    def add_common_ancestor(self, head, ploid):
        """
        Adds the specified ancestor (represented by the head of a segment
        chain) to the list of ancestors that find a common ancestor in
        the specified ploid of this individual.
        """
        heapq.heappush(self.common_ancestors[ploid], (head.left, head))


@dataclasses.dataclass
class DTWFSimulator(ancestry.SuperSimulator):
    U: float = 0.25e-1  # number of mutations per generation per ind
    s: float = 0.01

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        num_sig = 2  # number of standard deviation for distribution
        self.mean_load = self.U * (1 - self.s) / self.s
        self.num_fitness_classes = 2 * math.ceil(num_sig * math.sqrt(self.mean_load))
        self.min_fitness = max(0, self.mean_load - self.num_fitness_classes // 2)
        self.lineages = []
        self.p = self.expected_parental_distribution()
        self.parents_range = self.get_parents_range()

    def ancestors_remain(self):
        return True

    def verify(self):
        return True

    def expected_parental_distribution(self):
        exp_bps = int(1 / (self.L * self.r))
        num_segs = exp_bps + 1
        p = (math.floor(num_segs / 2) + 1) / num_segs
        return p

    def adjust_fitness_class(self, k):
        k -= self.min_fitness
        return int(max(0, min(self.num_fitness_classes - 1, k)))

    def get_ind_range(self):
        return

    def insert_lineage(self, lineage):
        self.lineages.append(lineage)
        self.num_lineages += 1

    def remove_lineage(self, lineage_id):
        lin = self.lineages.pop(lineage_id)
        self.num_lineages -= 1
        return lin

    def gen_parents_k(self, pheno):
        found = False
        while not found:
            i, j = self.rng.poisson(self.mean_load, size=2)
            value = self.p * i + (1 - self.p) * j
            found = abs(pheno - value) < 0.5
        return i, j

    def get_parents_range(self):
        ret = np.arange(self.min_fitness, self.min_fitness + self.num_fitness_classes)
        for i in range(ret.size):
            ret[i] = utils.poisson_pmf(ret[i], self.mean_load)
        ret /= np.sum(ret)
        return np.ceil(np.cumsum(ret) * self.Ne)

    def generation(self, nodes, tables):
        offspring = collections.defaultdict(list)
        for anc in self.lineages:
            # loose mutation with prob self.s * anc.value
            u = self.rng.uniform()
            if anc.value > 0:
                if u < self.s * (anc.value + self.min_fitness):
                    anc.value = 1

            parents_k = self.gen_parents_k(anc.value)
            for ploid in range(self.ploidy):
                pk = self.adjust_fitness_class
                start = 0
                if pk > 0:
                    start = self.parents_range[pk - 1]
                stop = self.parents_range[pk]
            # choose a parent from klass parents_k[ploid]

            # parent consists of two lineages with fitness k and k'
            # parent idxs range from [[0, ne * nh0], ..., [Ne - Ne * nhk, Ne]]
            parent = -1
            if parent not in offspring:
                offspring[parent] = []
            offspring[parent].append(anc)

        for children in offspring:
            H = [[], []]
            for child in children:
                # recombine across both parents
                # add recombined lineages if not None
                segs_pair = []  # generate by recombination
                for seg in segs_pair:
                    self.insert_lineage(seg)

                # collect segments inherited by the same individual
                for seg in segs_pair:
                    continue

            for ploid, h in enumerate(H):
                segments_to_merge = len(h)
                if segments_to_merge == 1:
                    h = []
                else:
                    c = ancestry.Lineage(len(nodes), [])
                    for interval, intersecting_lineages in ancestry.merge_ancestry(h):
                        c.ancestry.append(interval)
                        for lineage in intersecting_lineages:
                            tables.edges.add_row(
                                interval.left, interval.right, c.node, parent.node
                            )

    def generate_breakpoint(self):
        pass

    def recombine(self):
        pass

    def run(self, simplify=True, debug=False):
        return self._sim(simplify, debug)

    def _sim(self, simplify, debug=False):
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        nodes = []
        rng = np.random.default_rng(self.rng.randrange(2**16))

        t = 0
        while self.ancestors_remain():
            t += 1
            self.generation(nodes, tables)

        assert self.verify()

        return self.finalise(tables, nodes, simplify)
