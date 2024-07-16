import collections
import dataclasses
import math
import msprime
import numpy as np
import random
import tskit

from msbs import ancestry
from msbs import utils


@dataclasses.dataclass
class Ratchet:
    click_rate: float = 0.0
    start_value: int = 0


@dataclasses.dataclass
class Simulator(ancestry.SuperSimulator):
    U: float = 2e-3
    s: float = 1e-3
    bounded: bool = False

    def __post_init__(self):
        self.mean_load = self.U * (1 - self.s) / self.s
        self.num_lineages = 0
        self.min_fitness = 0
        self.rng = np.random.default_rng(self.seed)
        self.lineages = []
        self.bound = math.inf
        if self.bounded:
            # use distribution here??
            self.bound = 1 / self.s
            raise NotImplementedError
        self.num_coal_events = 0

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
        return self.num_lineages == 0

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

    def set_post_rec_fitness_class(self, value, p):
        start_value = value
        left_value = self.rng.binomial(start_value, p)
        # assign the remaining mutations to the right lineage
        right_value = start_value - left_value
        assert right_value >= 0
        left_value += self.rng.poisson(self.mean_load * (1 - p))
        right_value += self.rng.poisson(self.mean_load * p)
        return left_value, right_value

    def run(self, simplify=True, ca_events=False, end_time=None):
        initial_state = self._initial_setup(ca_events=ca_events, end_time=end_time)
        ts = self._complete(initial_state)
        if simplify:
            ts = ts.simplify()
        return ts

    def _complete(self, ts):
        # see Nicolaisen and Desai 2013
        rescale = np.exp(-self.U / self.s)
        # TO DO: look into this issue !!!
        if self.Ne * rescale < 1.0:
            R = (self.r * self.L)
            rescale = np.exp(-self.U / (self.s + R / 2))
        return msprime.sim_ancestry(
            initial_state=ts,
            recombination_rate=self.r,
            population_size=self.Ne * rescale,
            ploidy=self.ploidy,
        )

    def record_edges(self, lin, t, tables, nodes):
        for interval in lin.ancestry:
            tables.edges.add_row(interval.left, interval.right, len(nodes), lin.node)
        nodes.append(ancestry.Node(time=t))

    def common_ancestor_waiting_time_from_rate(self, rate):
        u = self.rng.expovariate(rate)
        return self.ploidy * self.Ne * u


@dataclasses.dataclass
class ZeroClassSimulator(Simulator):
    ratchet: Ratchet = Ratchet()

    def _initial_setup(
        self, simplify=False, debug=False, ca_events=False, end_time=None
    ):
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.populations.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.populations.add_row()
        tables.time_units = "generations"
        rng = random.Random(self.rng.integers(1))
        nodes = []
        if end_time is None:
            end_time = math.inf
        t = 0
        end_time = min(end_time, self.bound)

        for _ in range(self.n * self.ploidy):
            segment_chain = [ancestry.AncestryInterval(0, self.L, 1)]
            k = self.rng.poisson(self.mean_load)
            if k > self.min_fitness:
                self.insert_lineage(ancestry.Lineage(len(nodes), segment_chain, k))
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
            # we should only see lineages that are still carrying mutations
            assert total_num_muts >= len(num_muts)
            mu_rate = total_num_muts * self.s
            t_mu = math.inf if mu_rate == 0 else self.rng.exponential(1 / mu_rate)
            coal_rate = (
                self.num_lineages * (self.num_lineages - 1) / 2 if ca_events else 0.0
            )
            # should we be using a rescale factor here?
            rescale = 1.0
            # rescale = np.exp(-self.U / self.s + self.r * self.L / 2)
            t_ca = (
                math.inf
                if coal_rate == 0
                else self.rng.exponential(1 / coal_rate)
                * self.ploidy
                * self.Ne
                * rescale
            )
            t_click = (
                math.inf
                if self.ratchet.click_rate == 0.0
                else self.rng.exponential(1 / (self.ratchet.click_rate * self.U))
            )
            t_inc = min(t_mu, t_re, t_ca, t_click)
            if end_time < t_inc + t:
                t = end_time
                # take care of all floating lineages
                for lin in self.lineages:
                    self.record_edges(lin, t, tables, nodes)
                break

            t += t_inc
            if t_inc == t_re:  # recombination event
                idx = rng.choices(range(self.num_lineages), weights=lineage_links)[0]
                left_lineage = self.lineages[idx]
                breakpoint = self.rng.integers(
                    left_lineage.left + 1, left_lineage.right
                )
                assert left_lineage.left < breakpoint < left_lineage.right
                right_lineage = left_lineage.split(breakpoint)
                assert right_lineage.value == left_lineage.value
                # set values of left and right lineages
                p = breakpoint / self.L
                lvalue, rvalue = self.set_post_rec_fitness_class(left_lineage.value, p)
                left_lineage.value = lvalue
                right_lineage.value = rvalue
                if left_lineage.value <= self.min_fitness:
                    _ = self.remove_lineage(idx)
                    self.record_edges(left_lineage, t, tables, nodes)
                if right_lineage.value <= self.min_fitness:
                    self.record_edges(right_lineage, t, tables, nodes)
                else:
                    self.insert_lineage(right_lineage)

            elif t_inc == t_mu:  # decrement mutations
                # pick random idx
                idx = rng.choices(range(self.num_lineages), weights=num_muts)[0]
                # decrement
                self.lineages[idx].value -= 1
                # after decrementing
                if self.lineages[idx].value == self.min_fitness:
                    lin = self.remove_lineage(idx)
                    self.record_edges(lin, t, tables, nodes)
            elif t_inc == t_click:  # move ratchet
                self.min_fitness += 1
                for idx in range(self.num_lineages - 1, -1, -1):
                    if self.lineages[idx].value <= self.min_fitness:
                        lin = self.remove_lineage(idx)
                        assert lin.value <= self.min_fitness
                        self.record_edges(lin, t, tables, nodes)
            else:  # common ancestor event
                self.num_coal_events += 1
                a = self.remove_lineage(self.rng.integers(self.num_lineages))
                b = self.remove_lineage(self.rng.integers(self.num_lineages))
                k = math.ceil((a.value + b.value) / 2)
                c = ancestry.Lineage(len(nodes), [], k)
                for interval, intersecting_lineages in ancestry.merge_ancestry([a, b]):
                    # only add interval back into state if not ancestral to all samples
                    if interval.ancestral_to < self.n * self.ploidy:
                        c.ancestry.append(interval)
                    for lineage in intersecting_lineages:
                        tables.edges.add_row(
                            interval.left, interval.right, c.node, lineage.node
                        )

                nodes.append(ancestry.Node(time=t))
                if len(c.ancestry) > 0 and k > self.min_fitness:
                    self.insert_lineage(c)

        return self.finalise(tables, nodes, simplify)


@dataclasses.dataclass
class OGZeroClassSimulator(Simulator):
    def __post_init__(self):
        super().__post_init__()
        num_sig = 2
        self.num_fitness_classes = 2 * math.ceil(num_sig * math.sqrt(self.mean_load))
        self.min_fitness = max(0, self.mean_load - self.num_fitness_classes // 2)
        self.Q = self.generate_q()
        self.B_inv = np.linalg.inv(np.flip(self.Q)[:-1, :-1])

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

    def _initial_setup(
        self, simplify=False, debug=False, ca_events=False, end_time=None
    ):
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
                        lvalue, rvalue = self.set_post_rec_fitness_class(
                            left_lineage.value + self.min_fitness, p
                        )
                        left_lineage.value = self.adjust_fitness_class(lvalue)
                        right_lineage.value = self.adjust_fitness_class(rvalue)
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


@dataclasses.dataclass
class ZeroClassEmulator(Simulator):
    num_classes: int = 10

    def __post_init__(self):
        super().__post_init__()
        self.d = self._demography_factory()

    def reset(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.lineages = []

    def run(self, simplify=True):
        return self._sim(simplify=simplify)

    def _demography_factory(self):
        d = msprime.Demography()
        for pop_idx in range(10):
            nhk = utils.poisson_pmf(pop_idx, self.mean_load)
            d.add_population(initial_size=self.Ne * 2 * nhk)
            if pop_idx > 0:
                d.set_migration_rate(
                    source=pop_idx, dest=pop_idx - 1, rate=self.s * pop_idx
                )
        return d

    def _sim(self, simplify=True, **_):
        p = np.array(
            [utils.poisson_pmf(i, self.mean_load) for i in range(self.num_classes)]
        )
        p /= np.sum(p)
        sample_distr = self.rng.multinomial(self.n * self.ploidy, p)
        samples = {
            i: sample_distr[i] for i in range(sample_distr.size) if sample_distr[i] > 0
        }
        return msprime.sim_ancestry(
            samples,
            demography=self.d,
            sequence_length=self.L,
            recombination_rate=self.r,
            ploidy=1,
        )
