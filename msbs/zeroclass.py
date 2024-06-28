import collections
import dataclasses
import math
import numpy as np
import tskit

from msbs import ancestry
from msbs import utils


@dataclasses.dataclass
class Simulator(ancestry.SuperSimulator):
    U: float = 2e-3
    s: float = 1e-3
    
    def __post_init__(self):
        num_sig = 2
        self.mean_load = self.U * (1 - self.s) / self.s
        self.num_fitness_classes = 2 * math.ceil(num_sig * math.sqrt(self.mean_load))
        self.num_lineages = np.zeros(self.num_fitness_classes, dtype=np.int64)
        self.min_fitness = max(0, self.mean_load - self.num_fitness_classes // 2)
        self.Q = self.generate_q()
        self.rng = np.random.default_rng(12)
        self.lineages = []
        self.B_inv = np.linalg.inv(np.flip(self.Q)[:-1, :-1])

    def ancestors_remain(self):
        return np.sum(self.num_lineages) > 0

    def finalise(self, tables, nodes, simplify):
        for node in nodes:
            tables.nodes.add_row(
                flags=node.flags, time=node.time, metadata=node.metadata
            )
        tables.sort()
        tables.edges.squash()
        print(tables)
        ts = tables.tree_sequence()
        if simplify:
            ts = ts.simplify()

        return ts

    def reset(self, seed=None):
        self.seed = seed
        self.__post_init__()

    def adjust_fitness_class(self, k):
        k -= self.min_fitness
        return int(max(0, min(self.num_fitness_classes - 1, k)))

    def generate_q(self, num_sig=2):
        """
        Generates rate matrix to transition from class i to j.
        Gaining or losing a mutation happens at rate s
        """
        # determine expected variance in k
        Q = np.zeros((self.num_fitness_classes, self.num_fitness_classes))
        Q += np.eye(self.num_fitness_classes, k=-1)
        Q *= self.s * (np.arange(1, self.num_fitness_classes + 1) + self.min_fitness)
        Q[np.diag_indices(Q.shape[0])] = -np.sum(Q, axis=1)

        return Q

    def _intial_setup(self, simplify=True, debug=False):
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        nodes = []
        for _ in range(self.n * self.ploidy):
            segment_chain = [ancestry.AncestryInterval(0, self.L, 1)]
            k = self.adjust_fitness_class(self.rng.poisson(self.mean_load))
            self.lineages.append(ancestry.Lineage(len(nodes), segment_chain, k))
            nodes.append(ancestry.Node(time=0, flags=tskit.NODE_IS_SAMPLE))
        mu_free_times = utils.markov_chain_expected_absorption(self.B_inv)
        
        d = collections.deque()
        for lineage in self.lineages:
            t = 0
            d.append((t, lineage))
            while d:
                t, lineage = d.popleft()
                z = None
                re_rate = lineage.num_recombination_links * self.r
                k = lineage.value
                if k > 0:
                    assert k < self.num_fitness_classes
                    t_mu = mu_free_times[k]
                    t_re = (
                        math.inf if re_rate == 0 else self.rng.exponential(1 / re_rate)
                    )
                    t_inc = min(t_mu, t_re)
                    t += t_inc
                    if t_inc == t_re:  # recombination_event
                        left_lineage = lineage
                        breakpoint = self.rng.integers(
                            left_lineage.left + 1, left_lineage.right
                        )
                        assert left_lineage.left < breakpoint < left_lineage.right
                        right_lineage = left_lineage.split(breakpoint)
                        # set values of left and right lineages
                        left_lineage.value = 0
                        right_lineage.value = 0
                        d.append((t, left_lineage))
                        d.append((t, right_lineage))
                    else:  # loose all mutations
                        lineage.value = 0
                        z = lineage
                else:
                    if t != 0:
                        # add into initial_state immediately
                        z = lineage

                if z is not None:
                    # add into intial state
                    for interval in z.ancestry:
                        tables.edges.add_row(
                            interval.left, interval.right, len(nodes), z.node
                        )
                    nodes.append(ancestry.Node(time=t))

        return self.finalise(tables, nodes, simplify)
