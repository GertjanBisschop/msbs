import itertools
import random
import math
import dataclasses
import numpy as np
import tskit

from msbs import ancestry
from msbs import bins


@dataclasses.dataclass
class ZLineage(ancestry.Lineage):
    mutations: np.ndarray = None
    breakpoints: np.ndarray = None

    def __post_init__(self):
        self.freq = 0.0

    def set_fitness(self):
        assert self.mutations is not None
        assert self.breakpoints is not None
        self.value = np.sum(self.mutations)
        assert self.value >= 0
        self.freq = self.prob_i()

    def prob_i(self):
        fact = 1.0
        width = 1.0
        L = self.breakpoints[-1]
        assert L > 0
        for i in range(self.mutations.size):
            fact *= math.factorial(self.mutations[i])
            width *= ((self.breakpoints[i + 1] - self.breakpoints[i]) / L) ** (
                self.mutations[i]
            )

        ret = math.factorial(self.value) / fact * width
        assert ret <= 1
        return ret

    def coalescing(self, other):
        new_breakpoints = np.zeros(
            self.breakpoints.size + other.breakpoints.size, dtype=np.float64
        )
        new_mutations = np.zeros(new_breakpoints.size, dtype=np.int64)

        i = 0
        j = 0
        k = 0
        coalescing = True
        other_mutations = other.mutations.copy()
        self_mutations = self.mutations.copy()

        while (i < self.mutations.size) or (j < other_mutations.size):
            x_tmp = 0
            y_tmp = 0
            if self.breakpoints[i + 1] > other.breakpoints[j + 1]:
                x_tmp = other.breakpoints[j + 1]
                y_tmp = other_mutations[j]
                self_mutations[i] -= y_tmp
                j += 1
            elif self.breakpoints[i + 1] == other.breakpoints[j + 1]:
                x_tmp = other.breakpoints[j + 1]
                if self_mutations[i] == other_mutations[j]:
                    y_tmp = other_mutations[j]
                else:
                    y_tmp = -1
                i += 1
                j += 1
            else:
                x_tmp = self.breakpoints[i + 1]
                y_tmp = self_mutations[i]
                other_mutations[j] -= y_tmp
                i += 1

            if y_tmp < 0:
                coalescing = False
                break
            new_breakpoints[k + 1] = x_tmp
            new_mutations[k] = y_tmp
            k += 1

        if coalescing:
            new_mutations = new_mutations[:k]
            new_breakpoints = new_breakpoints[: k + 1]
            A = ZLineage(
                -1, None, np.sum(new_mutations), new_mutations, new_breakpoints
            )
        else:
            A = None

        return coalescing, A

    def split(self, bp, mean_mut, rng):

        left_ancestry = []
        right_ancestry = []

        for interval in self.ancestry:
            if interval.right <= bp:
                left_ancestry.append(interval)
            elif interval.left >= bp:
                right_ancestry.append(interval)
            else:
                assert interval.left < bp < interval.right
                left_ancestry.append(dataclasses.replace(interval, right=bp))
                right_ancestry.append(dataclasses.replace(interval, left=bp))
        self.ancestry = left_ancestry

        ## modify mutation counts left and right of breakpoint
        # identify breakpoint bin for bp
        right_lin = ZLineage(
            self.node,
            right_ancestry,
            -1,
            np.zeros(self.breakpoints.size, dtype=np.float64),
            np.zeros(self.mutations.size + 1, dtype=np.int64),
        )
        L = self.breakpoints[-1]
        i = 0

        while i < self.breakpoints.size:
            if self.breakpoints[i] < bp:
                i += 1
            else:
                new_breakpoint = bp != self.breakpoints[i]
                break
        width = self.breakpoints[i] - self.breakpoints[i - 1]
        if new_breakpoint:
            to_divide = self.mutations[i - 1]
            p = (bp - self.breakpoints[i - 1]) / width
            mut_left = 0
            if to_divide > 0:
                mut_left = rng.binomial(to_divide, p=p)
            self.mutations[i - 1] = mut_left
            right_lin.breakpoints[0] = bp
            assert mut_left <= to_divide
            right_lin.mutations[0] = to_divide - mut_left

        k = i
        j = 1
        while i < self.breakpoints.size:
            right_lin.breakpoints[j] = self.breakpoints[i]
            if i < self.mutations.size:
                right_lin.mutations[j] = self.mutations[i]
            i += 1
            j += 1

        # adjust left
        self.breakpoints = self.breakpoints[: k + 1]
        self.breakpoints[-1] = bp
        self.breakpoints = np.append(self.breakpoints, np.array(L))
        new_mutations = rng.poisson((L - bp) / L * mean_mut)
        self.mutations = self.mutations[:k]
        self.mutations = np.append(self.mutations, np.array(new_mutations))

        # adjust right
        right_lin.breakpoints = right_lin.breakpoints[int(~new_breakpoint) : j]
        right_lin.mutations = right_lin.mutations[int(~new_breakpoint) : j - 1]
        right_lin.breakpoints = np.append(np.zeros(1), right_lin.breakpoints)
        right_lin.mutations = np.append(np.zeros(1), right_lin.mutations)
        right_lin.mutations[0] = rng.poisson(bp / L * mean_mut)
        if not new_breakpoint:
            right_lin.mutations[-1] = rng.poisson(width / L * mean_mut)

        assert self.mutations.size == (self.breakpoints.size - 1)
        assert right_lin.mutations.size == (right_lin.breakpoints.size - 1)
        self.set_fitness()
        right_lin.set_fitness()

        return right_lin


@dataclasses.dataclass
class ZSimulator(ancestry.SuperSimulator):
    U: float = 0.25e-1  # number of mutations per generation per ind
    s: float = 0.01

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.lineages = []
        self.num_lineages = 0

    def print_state(self):
        for lin in self.lineages:
            print(lin, lin.mutations, lin.value)

    def remove_lineage(self, lineage_id):
        lin = self.lineages.pop(lineage_id)
        self.num_lineages -= 1
        return lin

    def insert_lineage(self, lineage):
        self.lineages.append(lineage)
        lineage.set_fitness()
        self.num_lineages += 1

    def mutation_event(self, total_mass):
        random_mass = self.rng.random()
        observed_mass = 0.0
        for lineage in self.lineages:
            observed_mass += lineage.value / total_mass
            if observed_mass > random_mass:
                break

        assert lineage.value > 0
        assert np.sum(lineage.mutations > 0)
        random_index = self.rng.choices(
            range(lineage.mutations.size), weights=lineage.mutations
        )[0]
        lineage.mutations[random_index] -= 1
        lineage.value -= 1
        assert np.all(lineage.mutations >= 0)

    def pairwise_coal_rate(self, child, sib, mean_mut):
        ret = 0.0
        assert child.freq > 0.0
        assert sib.freq > 0.0
        child_value = child.value
        if child.value == sib.value:
            coal_bool, parent = child.coalescing(sib)
            if coal_bool:
                parent_freq = parent.prob_i()
                f_i = (
                    np.exp(-mean_mut)
                    * (mean_mut) ** child.value
                    / math.factorial(child.value)
                )
                ret = parent_freq / (f_i * child.freq * sib.freq)

        return ret / (self.ploidy * self.Ne)

    def get_coal_rate(self, mean_mut):
        coal_rates = np.zeros(math.comb(self.num_lineages, 2))
        for pair in itertools.combinations(range(self.num_lineages), 2):
            a, b = pair
            coal_rates[bins.combinadic_map(pair)] = self.pairwise_coal_rate(
                self.lineages[a], self.lineages[b], mean_mut
            )

        return coal_rates

    def common_ancestor_waiting_time_from_rate(self, rate):
        if rate == 0:
            return math.inf
        u = self.rng.expovariate(rate)
        return u

    def common_ancestor_event(self, coal_rates, tables, node_id):
        i = self.rng.choices(range(coal_rates.size), weights=coal_rates)[0]
        ai, bi = bins.reverse_combinadic_map(i)
        a = self.remove_lineage(ai)
        b = self.remove_lineage(bi)
        coal_bool, c = a.coalescing(b)
        assert coal_bool
        c.node = node_id
        c.ancestry = []
        for interval, intersecting_lineages in ancestry.merge_ancestry([a, b]):
            c.ancestry.append(interval)
            for lineage in intersecting_lineages:
                tables.edges.add_row(
                    interval.left, interval.right, c.node, lineage.node
                )
        self.insert_lineage(c)

        return c

    def test_lineages(self):
        for lin in self.lineages:
            if lin.value != np.sum(lin.mutations):
                return False

        return True

    def run(self, simplify=True):
        return self._sim(simplify)

    def _sim(self, simplify):
        rng = np.random.default_rng(self.rng.randint(1, 2**16))
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        nodes = []
        mean_load = self.U / self.s  # lambda_g
        mean_preload = self.U * (1 - self.s) / self.s  # lambda_f
        for _ in range(self.n * self.ploidy):
            segment_chain = [ancestry.AncestryInterval(0, self.L, 1)]
            mutations = rng.poisson(mean_load, size=1)
            breakpoints = np.array([0, self.L])
            lineage = ZLineage(
                len(nodes), segment_chain, np.sum(mutations), mutations, breakpoints
            )
            lineage.set_fitness()
            self.insert_lineage(lineage)
            nodes.append(ancestry.Node(time=0, flags=tskit.NODE_IS_SAMPLE))

        t = 0
        while not super().stop_condition():
            lineage_links = [
                lineage.num_recombination_links for lineage in self.lineages
            ]
            total_links = sum(lineage_links)
            re_rate = total_links * self.r
            t_re = math.inf if re_rate == 0 else self.rng.expovariate(re_rate)
            coal_rates = self.get_coal_rate(mean_preload)
            t_ca = self.common_ancestor_waiting_time_from_rate(np.sum(coal_rates))
            num_mutations = sum(lin.value for lin in self.lineages)
            t_mut = (
                math.inf
                if num_mutations == 0
                else self.rng.expovariate(self.s * num_mutations)
            )
            t_inc = min(t_re, t_ca, t_mut)
            t += t_inc

            if t_inc == t_re:  # recombination
                # print("----------recombination-----------")
                left_lineage = self.rng.choices(self.lineages, weights=lineage_links)[0]
                breakpoint = self.rng.randrange(
                    left_lineage.left + 1, left_lineage.right
                )
                assert left_lineage.left < breakpoint < left_lineage.right
                right_lineage = left_lineage.split(breakpoint, mean_load, rng)
                self.insert_lineage(right_lineage)
                child = left_lineage.node
                assert right_lineage.node == child
            elif t_inc == t_ca:  # common ancestor event
                # print("---------ca_event---------")
                _ = self.common_ancestor_event(coal_rates, tables, len(nodes))
                nodes.append(ancestry.Node(time=t))

            else:  # mutation
                # print("---------mutation---------")
                self.mutation_event(num_mutations)

        return self.finalise(tables, nodes, simplify)
