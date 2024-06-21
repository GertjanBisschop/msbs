import collections
import random
import math
import dataclasses
import numpy as np
import tskit

from scipy.linalg import expm
from typing import Callable

from msbs import ancestry
from msbs import utils


def fk(Q: np.ndarray, I: np.ndarray, k: int, num_lins_vec: np.ndarray) -> Callable:
    """
    Returns a function that gives the instantaneous coalescence
    rate at time `time` in class `k` given the rate matrix `Q`
    and the number of lineages `num_lins_vec` present in
    each fitness class.
    """

    def _fk(time: float) -> float:
        freqs = I @ expm(Q * time)
        freqs_k = np.sum(num_lins_vec * freqs[:, k])

        return max(0, freqs_k * (freqs_k - 1) / 2)

    return _fk


@dataclasses.dataclass
class Simulator(ancestry.SuperSimulator):
    K: ancestry.FitnessClassMap = None
    U: float = 0.25e-1  # number of mutations per generation per ind
    s: float = 0.01

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        if self.K is None:
            position = np.zeros(2)
            position[-1] = self.L
            self.K = ancestry.FitnessClassMap(position, np.zeros(1))
        num_sig = 2  # number of standard deviation for distribution
        self.mean_load = self.U * (1 - self.s) / self.s
        self.num_fitness_classes = 2 * math.ceil(num_sig * math.sqrt(self.mean_load))
        self.num_lineages = np.zeros(self.num_fitness_classes, dtype=np.int64)
        self.min_fitness = max(0, self.mean_load - self.num_fitness_classes // 2)
        self.Q = self.generate_q()
        self.coal_rate_fs = [fk(self.Q, i) for i in range(self.Q.shape[0])]
        self.lineages = []
        self.info = collections.defaultdict(list)

    def __str__(self):
        return (
            super().__str__()
            + f"\nmin fitness: {self.min_fitness}, num classes: {self.num_fitness_classes}"
        )

    def print_state(self, last_event):
        print(f"------------{last_event} event-------------")
        print(self.num_lineages)
        for lineage in self.lineages:
            print(lineage)
        print("-----------------------------------")

    def verify(self, last_event):
        no_error = True
        if np.any(self.num_lineages < 0):
            no_error = False
        test = np.zeros_like(self.num_lineages)
        for lin in self.lineages:
            test[lin.value] += 1
            no_error = np.array_equal(test, self.num_lineages)
        if not no_error:
            self.print_state(last_event)
        return no_error

    def common_ancestor_waiting_time_from_rate(self, rate):
        if rate == 0.0:
            return math.inf
        u = self.rng.expovariate(rate)
        return self.ploidy * self.Ne * u

    def generate_q(self):
        """
        Generates rate matrix to transition from class i to j.
        Gaining or losing a mutation happens at rate s
        """
        Q = np.zeros((self.num_fitness_classes, self.num_fitness_classes))
        Q += np.eye(self.num_fitness_classes, k=-1)
        # Q_ij = i * s
        Q *= self.s * (np.arange(1, self.num_fitness_classes + 1) + self.min_fitness)
        Q[np.diag_indices(Q.shape[0])] = -np.sum(Q, axis=1)

        return Q

    def remove_lineage(self, lineage_id):
        lin = self.lineages.pop(lineage_id)
        self.num_lineages[lin.value] -= 1
        return lin

    def remove_lineage_within_class(self, class_id):
        mask = [lin.value == class_id for lin in self.lineages]
        lin_idx = self.rng.choices(range(len(self.lineages)), weights=mask)[0]
        lin = self.lineages.pop(lin_idx)
        self.num_lineages[lin.value] -= 1
        return lin

    def insert_lineage(self, lineage):
        self.lineages.append(lineage)
        self.num_lineages[lineage.value] += 1

    def adjust_fitness_class(self, k):
        k -= self.min_fitness
        return int(max(0, min(self.num_fitness_classes - 1, k)))

    def stop_condition(self):
        return super().stop_condition()

    def run(self, simplify=True, debug=False):
        return self._sim(simplify, debug)

    def _sim(self, simplify, debug):
        """
        Experimental implementation of coalescent with local Ne map along genome.

        NOTE! This hasn't been statistically tested and is probably not correct.
        """
        if debug:
            print(self)
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        nodes = []
        rng = np.random.default_rng(self.rng.randrange(2**16))
        I = np.eye(sim.Q.shape[0])
        load_g = self.U / self.s
        hk_probs = np.zeros(self.Q.shape[0])
        hk_probs /= np.sum(hk_probs)
        for k in range(hk_probs.size):
            hk_probs = utils.poisson_pmf(k, load_g)
        last_event = "in"
        for _ in range(self.n * self.ploidy):
            segment_chain = [ancestry.AncestryInterval(0, self.L, 1)]
            k = self.adjust_fitness_class(rng.poisson(self.mean_load))
            self.insert_lineage(ancestry.Lineage(len(nodes), segment_chain, k))
            nodes.append(ancestry.Node(time=0, flags=tskit.NODE_IS_SAMPLE))

        t = 0
        while not self.stop_condition():
            if debug:
                self.print_state(last_event)
            lineage_links = [
                lineage.num_recombination_links for lineage in self.lineages
            ]
            total_links = sum(lineage_links)
            re_rate = total_links * self.r
            t_re = math.inf if re_rate == 0 else self.rng.expovariate(re_rate)
            t_ca = math.inf
            coal_rate_fs = [
                fk(self.Q, I, i, self.num_lineages) for i in range(self.Q.shape[0])
            ]
            ca_class = None
            for idx in range(self.num_fitness_classes):
                # draw waiting time for fitness class `idx`
                temp = utils.sample_nhpp(coal_rate_fs[idx], self.rng)
                k = idx + self.min_fitness
                temp *= self.ploidy * self.Ne * hk_probs[idx]
                if temp < t_ca:
                    t_ca = temp
                    ca_class = idx

            t_inc = min(t_re, t_ca)
            delta_t = t_inc
            t += t_inc

            if t_inc == t_re:  # recombination
                last_event = "re"
                left_lineage = self.rng.choices(self.lineages, weights=lineage_links)[0]
                breakpoint = self.rng.randrange(
                    left_lineage.left + 1, left_lineage.right
                )
                assert left_lineage.left < breakpoint < left_lineage.right
                # adjust fitness class left and right lineage
                left_av = self.K.weighted_average(0.0, left_lineage.right, total=True)
                # fitness class value is normalised
                self.num_lineages[left_lineage.value] -= 1
                left_lin_k = left_lineage.value + self.min_fitness
                p = left_av / self.K.av
                k = rng.binomial(left_lin_k, p=p)
                right_lineage = left_lineage.split(breakpoint)
                # adjust fitness class of new left and right segments post rec
                left_lineage.value = k
                left_lineage.value += rng.poisson(self.mean_load * (1 - p))
                left_lineage.value = self.adjust_fitness_class(left_lineage.value)
                self.num_lineages[left_lineage.value] += 1
                right_lineage.value = left_lin_k - k
                right_lineage.value += rng.poisson(self.mean_load * p)
                right_lineage.value = self.adjust_fitness_class(right_lineage.value)
                self.insert_lineage(right_lineage)
                child = left_lineage.node
                assert right_lineage.node == child

            else:  # common ancestor event
                last_event = "ca"
                # given ca_event in ca_population pick two random lineages a,b
                # that could have coalesced in that fitness class
                freqs = I @ expm(Q * delta_t)
                class_rate = self.num_lineages * freqs[:, ca_class]
                # sample a and b given weights in class_rate
                class_idxs = rng.choice(
                    np.arange(self.num_fitness_classes),
                    size=2,
                    p=class_rate / np.sum(class_rate),
                )
                a = self.remove_lineage_within_class(class_idxs[0])
                b = self.remove_lineage_within_class(class_idxs[1])
                c = ancestry.Lineage(len(nodes), [], ca_class)
                for interval, intersecting_lineages in ancestry.merge_ancestry([a, b]):
                    c.ancestry.append(interval)
                    for lineage in intersecting_lineages:
                        tables.edges.add_row(
                            interval.left, interval.right, c.node, lineage.node
                        )
                nodes.append(ancestry.Node(time=t))
                self.insert_lineage(c)

            assert self.verify(last_event)

        return self.finalise(tables, nodes, simplify)
