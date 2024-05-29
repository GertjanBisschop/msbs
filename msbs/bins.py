import itertools
import random
import math
import dataclasses
import numpy as np
import tskit

from msbs import ancestry


def combinadic_map(sorted_pair):
    """
    Maps a pair of indices to a unique integer.
    """
    return int((sorted_pair[0]) + sorted_pair[1] * (sorted_pair[1] - 1) / 2)


def reverse_combinadic_map(idx, k=2):
    """
    Maps a unique index to a unique pair.
    """
    while k > 0:
        i = k - 1
        num_combos = 0
        while num_combos <= idx:
            i += 1
            num_combos = math.comb(i, k)
        yield i - 1
        idx -= math.comb(i - 1, k)
        k -= 1


@dataclasses.dataclass
class BinLineage(ancestry.Lineage):
    def __post_init__(self):
        self.bins = None

    def __lt__(self, other):
        for i, j in zip(self.bins, other.bins):
            if i != j:
                return i < j

    def set_value(self):
        self.value = np.sum(self.bins)

    def set_bins(self, mean_muts, num_bins, rng):
        self.bins = rng.poisson(mean_muts, num_bins)

    def split(self, breakpoint, mean_muts, binwidth, rng):

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
        right_lin = BinLineage(self.node, right_ancestry)
        right_lin.bins = self.bins.copy()

        # modify bin counts left and right of breakpoint
        i = breakpoint // binwidth  # bin at breakpoint index
        value = self.bins[i]
        p = breakpoint % binwidth / binwidth
        q = (binwidth - breakpoint % binwidth) / binwidth
        value_left = rng.binomial(n=value, p=p)
        self.bins[i] = value_left + rng.poisson(mean_muts * q)
        j = self.bins.size - (i + 1)
        if j > 0:
            self.bins[i + 1 :] = rng.poisson(mean_muts, j)
        right_lin.bins[i] = value - value_left + rng.poisson(mean_muts * p)
        right_lin.bins[:i] = rng.poisson(mean_muts, i)

        return right_lin


@dataclasses.dataclass
class BinSimulator(ancestry.SuperSimulator):
    U: float = 2.5e-1  # number of mutations per generation per ind
    s: float = 0.01
    num_bins: int = 10

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.lineages = []
        self.num_lineages = 0
        self.bins = np.linspace(0, self.L, num=self.num_bins, endpoint=False)
        self.binwidth = self.L // self.num_bins
        assert self.num_bins * self.binwidth == self.L

    def remove_lineage(self, lineage_id):
        lin = self.lineages.pop(lineage_id)
        self.num_lineages -= 1
        return lin

    def insert_lineage(self, lineage):
        self.lineages.append(lineage)
        self.num_lineages += 1

    def mutation_event(self, total_mass):
        random_mass = self.rng.random()
        observed_mass = 0.0
        for lineage in self.lineages:
            observed_mass += lineage.value / total_mass
            if observed_mass > random_mass:
                break

        nonzero = np.nonzero(lineage.bins)[0]
        assert nonzero.size > 0
        binindex = nonzero[self.rng.randrange(nonzero.size)]
        lineage.bins[binindex] -= 1
        lineage.value -= 1

    def prob_i(self, lin):
        fact = np.prod([math.factorial(count) for count in lin.bins])
        ret = (
            math.factorial(lin.value)
            / fact
            * np.prod((self.binwidth / self.L) ** lin.bins)
        )
        assert ret <= 1
        return ret

    def pairwise_coal_rate(self, child, sib, mean_mut):
        """
        deviation from zeng et al. 2011, because of bins all prob_i's will
        be identical.
        """
        ret = 0.0
        if child.value == sib.value:
            if np.all(child.bins == sib.bins):
                f_i = (
                    np.exp(-mean_mut)
                    * (mean_mut) ** child.value
                    / math.factorial(child.value)
                )
                ret = 1 / (f_i * self.prob_i(child) * self.ploidy * self.Ne)
        return ret

    def get_coal_rate(self, mean_mut):
        coal_rates = np.zeros(math.comb(self.num_lineages, 2))
        for pair in itertools.combinations(range(self.num_lineages), 2):
            a, b = pair
            coal_rates[combinadic_map(pair)] = self.pairwise_coal_rate(
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
        ai, bi = reverse_combinadic_map(i)
        a = self.remove_lineage(ai)
        b = self.remove_lineage(bi)
        c = BinLineage(node_id, [], a.value)
        assert np.all(a.bins == b.bins)
        c.bins = a.bins.copy()
        for interval, intersecting_lineages in ancestry.merge_ancestry([a, b]):
            c.ancestry.append(interval)
            for lineage in intersecting_lineages:
                tables.edges.add_row(
                    interval.left, interval.right, c.node, lineage.node
                )
        self.insert_lineage(c)

        return c

    def run(self, simplify=True):
        return self._sim_bins(simplify)

    def _sim_bins(self, simplify):
        rng = np.random.default_rng(self.rng.randint(1, 2**16))
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        nodes = []
        mean_load = self.U / self.s / self.binwidth  # lambda_g / binwidth
        mean_preload = self.U / (1 - self.s)  # lambda_f
        for _ in range(self.n * self.ploidy):
            segment_chain = [ancestry.AncestryInterval(0, self.L, 1)]
            lineage = BinLineage(len(nodes), segment_chain)
            lineage.set_bins(mean_load, self.num_bins, rng)
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
            t_mut = self.rng.expovariate(self.s * num_mutations)
            t_inc = min(t_re, t_ca, t_mut)
            t += t_inc
            
            if t_inc == t_re:  # recombination
                #print("----------recombination-----------")
                left_lineage = self.rng.choices(self.lineages, weights=lineage_links)[0]
                breakpoint = self.rng.randrange(
                    left_lineage.left + 1, left_lineage.right
                )
                assert left_lineage.left < breakpoint < left_lineage.right
                right_lineage = left_lineage.split(
                    breakpoint, mean_load, self.binwidth, rng
                )
                self.insert_lineage(right_lineage)
                child = left_lineage.node
                assert right_lineage.node == child
            elif t_inc == t_ca:  # common ancestor event
                #print("---------ca_event---------")
                _ = self.common_ancestor_event(coal_rates, tables, len(nodes))
                nodes.append(ancestry.Node(time=t))

            else:  # mutation
                #print("---------mutation---------")
                self.mutation_event(num_mutations)

        return self.finalise(tables, nodes, simplify)
