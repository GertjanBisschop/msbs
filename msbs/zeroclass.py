import collections
import dataclasses
import math
import msprime
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
        self.rng = np.random.default_rng(self.seed)
        self.lineages = []
        self.B_inv = np.linalg.inv(np.flip(self.Q)[:-1, :-1])

    def ancestors_remain(self):
        return np.sum(self.num_lineages) > 0

    def finalise(self, tables, nodes, simplify):
        for node in nodes:
            tables.nodes.add_row(
                flags=node.flags, time=node.time, metadata=node.metadata, population=0
            )
        tables.sort()
        tables.edges.squash()
        tables.sort()
        ts = tables.tree_sequence()
        if simplify:
            ts = ts.simplify()

        return ts

    def stop_condition(self):
        return np.sum(self.num_lineages) == 0

    def reset(self, seed=None):
        self.seed = seed
        self.__post_init__()

    def remove_lineage(self, lineage_id):
        lin = self.lineages.pop(lineage_id)
        self.num_lineages -= 1
        return lin

    def insert_lineage(self, lineage):
        self.lineages.append(lineage)
        self.num_lineages += 1

    def adjust_fitness_class(self, k):
        k -= self.min_fitness
        return int(max(0, min(self.num_fitness_classes - 1, k)))

    def set_post_rec_fitness_class(self, value, p):
        start_value = value + self.min_fitness
        value = self.rng.binomial(start_value, p)
        value += self.rng.poisson(self.mean_load * (1 - p))
        return self.adjust_fitness_class(value)

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

    def run(self, simplify=True):
        initial_state = self._intial_setup(simplify=False)
        ts = self._complete(initial_state)
        if simplify:
            ts = ts.simplify()
        return ts

    def _intial_setup(self, simplify=True, debug=False):
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.populations.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.populations.add_row()
        tables.time_units = "generations"
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
                    t_mu = mu_free_times[k - 1]
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
                        p = breakpoint / self.L
                        left_lineage.value = self.set_post_rec_fitness_class(
                            left_lineage.value, p
                        )
                        right_lineage.value = self.set_post_rec_fitness_class(
                            right_lineage.value, 1 - p
                        )
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

    def _intial_setup_stepwise(self, simplify=True, debug=False):
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.populations.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.populations.add_row()
        tables.time_units = "generations"
        nodes = []
        for _ in range(self.n * self.ploidy):
            segment_chain = [ancestry.AncestryInterval(0, self.L, 1)]
            k = self.adjust_fitness_class(self.rng.poisson(self.mean_load))
            self.lineages.append(ancestry.Lineage(len(nodes), segment_chain, k))
            nodes.append(ancestry.Node(time=0, flags=tskit.NODE_IS_SAMPLE))

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
                    mu_rate = self.s * k
                    t_mu = self.rng.exponential(1 / mu_rate)
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
                        p = breakpoint / self.L
                        left_lineage.value = self.set_post_rec_fitness_class(
                            left_lineage.value, p
                        )
                        right_lineage.value = self.set_post_rec_fitness_class(
                            right_lineage.value, 1 - p
                        )
                        d.append((t, left_lineage))
                        d.append((t, right_lineage))
                    else:
                        lineage.value -= 1
                        d.append((t, lineage))
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

    def _intial_setup_stepwise_all(self, simplify=True, debug=False):
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.populations.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.populations.add_row()
        tables.time_units = "generations"
        nodes = []
        for _ in range(self.n * self.ploidy):
            segment_chain = [ancestry.AncestryInterval(0, self.L, 1)]
            k = self.adjust_fitness_class(self.rng.poisson(self.mean_load))
            self.lineages.append(ancestry.Lineage(len(nodes), segment_chain, k))
            nodes.append(ancestry.Node(time=0, flags=tskit.NODE_IS_SAMPLE))

        while not self.stop_condition():
            lineage_links = [
                lineage.num_recombination_links for lineage in self.lineages
            ]
            total_links = sum(lineage_links)
            re_rate = total_links * self.r
            t_re = math.inf if re_rate == 0 else self.rng.exponential(1 / re_rate)
            num_muts = [lineage.value for lineage in self.lineages]
            total_num_muts = sum(num_muts)
            mu_rate = total_num_muts * self.s
            t_mu = math.inf if mu_rate == 0 else self.rng.exponential(1 / mu_rate)
            t_inc = min(t_mu, t_re)
            t += t_inc

            if t_inc == t_re:  # recombination_event
                left_lineage = self.rng.choices(self.lineages, weights=lineage_links)[0]
                breakpoint = self.rng.integers(
                    left_lineage.left + 1, left_lineage.right
                )
                assert left_lineage.left < breakpoint < left_lineage.right
                right_lineage = left_lineage.split(breakpoint)
                # set values of left and right lineages
                p = breakpoint / self.L
                left_lineage.value = self.set_post_rec_fitness_class(
                    left_lineage.value, p
                )
                right_lineage.value = self.set_post_rec_fitness_class(
                    right_lineage.value, 1 - p
                )
                self.insert_lineage(right_lineage)

            else:  # decrement mutations
                # pick random idx
                idx = self.rng.choices(range(self.num_lineages), weights=self.num_muts)[
                    0
                ]
                # decrement
                self.lineages[idx] -= 1
                # after decrementing
                if self.lineages[idx].value == 0:
                    lin = self.remove_lineage(idx)
                    # insert lineage into state
                    for interval in lin.ancestry:
                        tables.edges.add_row(
                            interval.left, interval.right, len(nodes), lin.node
                        )
                    nodes.append(ancestry.Node(time=t))

        return self.finalise(tables, nodes, simplify)

    def _complete(self, ts):
        rescale = np.exp(-self.U / self.s)
        return msprime.sim_ancestry(
            initial_state=ts,
            recombination_rate=self.r,
            population_size=self.Ne * rescale,
            ploidy=self.ploidy,
        )
