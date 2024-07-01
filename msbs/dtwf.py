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
        self.lineages = []
        self.p0_p = self.expected_parental_distribution()

    def ancestors_remain(self):
        return True

    def verify(self):
        return True

    def expected_parental_distribution(self):
        # using self.r
        return 1.0

    def get_ind_range(self):
        return

    def losing_mut_prob(self, k_now, k_future):
        delta_k = k_now - k_future
        if delta_k < 0:
            return 0.0
        else:
            # compute prob of losing delta_k mutations in a single time step
            return self.s * k_now

    def pick_parent(self, child, rng, num_switches):
        u = 1.0
        acc_prob = 0.0
        parent_id = -1

        while u > acc_prob:
            # sample parent given nh_k distribution
            parent_k = self.rng.poisson(self.mean_load, size=2)
            # compute probability of child having this parent
            # given expected number of switches
            # contribution by each parent
            k_child = self.p0_p * parent_k[0] + (1 - self.p0_p) * parent_k[1]
            # compute probability of having lost/gained k_child - child.value mutations
            acc_prob = 1.0
            u = self.rng.random()

        return parent_id, parent_k

    def generation(self):
        offspring = collections.defaultdict(list)
        for anc in self.lineages:
            # choose a parent

            # parent consists of two lineages with fitness k and k'
            parent = None
            if parent not in offspring:
                offspring[parent] = []
            offspring[parent].append(anc)

        # Draw recombinations in children and sort segments by
        # inheritance direction.

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
            self.generation()

        assert self.verify()

        return self.finalise(tables, nodes, simplify)
