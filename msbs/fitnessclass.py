import collections
import random
import math
import dataclasses
import numpy as np
import tskit

from scipy.linalg import expm
from typing import List

from msbs import ancestry
from msbs import utils

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
        self.mean_load = self.U * (1 - self.s) / self.s
        self.Q = self.generate_q()
        self.num_fitness_classes = self.Q.shape[0]
        self.num_lineages = np.zeros(self.num_fitness_classes, dtype=np.uint64)
        self.min_fitness = max(0, self.mean_load - self.num_fitness_classes // 2)
        self.lineages = []
        self.lineage_dict = collections.defaultdict(List)

    def print_state(self):
        for lineage in self.lineages:
            print(lineage)
        print("------------------------")

    def common_ancestor_waiting_time_from_rate(self, rate, scaling=1.0):
        u = self.rng.expovariate(rate)
        return self.ploidy * self.Ne * u * scaling

    def generate_q(self, num_sig=2):
        """
        Generates rate matrix to transition from class i to j.
        Gaining or losing a mutation happens at rate s
        """
        num_fit_classes = 2 * math.ceil(num_sig * math.sqrt(self.mean_load))
        Q = np.zeros((num_fit_classes, num_fit_classes))
        Q += np.eye(num_fit_classes, k=-1)
        Q *= self.s
        Q[np.diag_indices(Q.shape[0])] = -np.sum(Q, axis=1)

        return Q

    def remove_lineage(self, lineage_id):
        lin = self.lineages.pop(lineage_id)
        self.num_lineages[lin.value] -= 1
        return lin

    def remove_lineage_within_class(self, class_id):
        mask = [lin.value == class_id for lin in self.lineages]
        lin = self.rng.choices(self.lineages, weights=mask)[0]
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

    def run(self, simplify=True):
        return self._sim(simplify)

    def _sim(self, simplify=True):
        """
        Experimental implementation of coalescent with local Ne map along genome.

        NOTE! This hasn't been statistically tested and is probably not correct.
        """

        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        nodes = []
        rng = np.random.default_rng(self.rng.randrange(2**16))
        load_g = self.U / self.s
        freqs = np.eye(self.num_fitness_classes)
        for _ in range(self.n * self.ploidy):
            segment_chain = [ancestry.AncestryInterval(0, self.L, 1)]
            k = self.adjust_fitness_class(rng.poisson(self.mean_load))
            self.insert_lineage(ancestry.Lineage(len(nodes), segment_chain, k))
            nodes.append(ancestry.Node(time=0, flags=tskit.NODE_IS_SAMPLE))

        t = 0
        # given rec_rate and num_lineages, expected time to first event in absence of bs is:
        delta_t = min(
            1 / (self.L * np.sum(self.num_lineages) * self.r),
            self.ploidy * self.Ne / math.comb(np.sum(self.num_lineages), 2),
        )
        while not self.stop_condition():
            lineage_links = [
                lineage.num_recombination_links for lineage in self.lineages
            ]
            total_links = sum(lineage_links)
            re_rate = total_links * self.r
            t_re = math.inf if re_rate == 0 else self.rng.expovariate(re_rate)
            # assuming for now that ca_rate is constant in between events
            # and informed by the previous time step
            # alternatively, we could treat ca_rate as non-homogeneous and draw
            # waiting time while updating the freqs vector.
            t_ca = math.inf
            # ideally we reset freqs at the beginning of the second loop / after first event
            freqs = expm(self.Q * delta_t) @ freqs.T
            ca_rate = np.zeros(self.num_fitness_classes)
            ca_class = None
            for idx in range(self.num_fitness_classes):
                ca_rate[idx] += np.sum(self.num_lineages * freqs[:, idx])
                k = idx + self.min_fitness
                #hk = utils.poisson_pmf(k, load_g) 
                hk = 1.0
                temp = self.common_ancestor_waiting_time_from_rate(ca_rate[idx], hk)
                if temp < t_ca:
                    t_ca = temp
                    ca_class = idx

            t_inc = min(t_re, t_ca)
            delta_t = t_inc
            t += t_inc

            if t_inc == t_re:  # recombination
                print("----------recombination-----------")
                left_lineage = self.rng.choices(self.lineages, weights=lineage_links)[0]
                breakpoint = self.rng.randrange(
                    left_lineage.left + 1, left_lineage.right
                )
                assert left_lineage.left < breakpoint < left_lineage.right
                # adjust fitness class left and right lineage
                left_av = self.K.weighted_average(0.0, left_lineage.right, total=True)
                # fitness class value is normalised
                left_lin_k = left_lineage.value + self.min_fitness
                p = left_av / self.K.av
                k = rng.binomial(left_lin_k, p=p) # error: n < 0
                right_lineage = left_lineage.split(breakpoint)
                right_lineage.value = left_lin_k - k
                left_lineage.value += rng.poisson(self.mean_load * (1 - p))
                left_lineage.value = self.adjust_fitness_class(left_lineage.value)
                right_lineage.value += rng.poisson(self.mean_load * p)
                right_lineage.value = self.adjust_fitness_class(right_lineage.value)
                self.insert_lineage(right_lineage)
                child = left_lineage.node
                assert right_lineage.node == child

            else:  # common ancestor event
                print("------------ca_event-------------")     
                # given ca_event in ca_population pick two random lineages a,b
                # that could have coalesced in that fitness class
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

        return self.finalise(tables, nodes, simplify)
