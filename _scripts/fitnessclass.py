import abc
import dataclasses
import itertools
import matplotlib.pyplot as plt
import msprime
import numpy as np
import os
import pathlib
import pyslim
import subprocess
import warnings
import tskit

from tqdm import tqdm
from typing import List

from msbs import ancestry
from msbs import fitnessclass
from msbs import zeroclass

_slim_executable = ["./_data/slim"]


@dataclasses.dataclass
class Stat:
    dim: int
    label: str

    @abc.abstractmethod
    def compute(self, ts: tskit.TreeSequence) -> float:
        return

    @abc.abstractmethod
    def plot(data: np.ndarray, outfile: pathlib.Path, models:List[str]) -> None:
        return
    
@dataclasses.dataclass
class SummaryStat(Stat):
    
    @abc.abstractmethod
    def compute(self, ts: tskit.TreeSequence) -> float:
        return

    def plot(self, data: np.ndarray, outfile: pathlib.Path, models:List[str]) -> None:
        plt.violinplot(data.tolist(), showmeans=True)
        plt.xticks(
            [y + 1 for y in range(data.shape[0])],
            labels=models + ["neutral", "neutral rescaled", "slim",],
        )
        plt.savefig(outfile, dpi=120)
        plt.close("all")


@dataclasses.dataclass
class CovStat(Stat):
    r: float

    @abc.abstractmethod
    def compute(self, ts: tskit.TreeSequence) -> float:
        return
    
    def expected_cov(self):
        return (self.r + 18) / (self.r**2 + 13 * self.r + 18)

    
    def plot(self, data: np.ndarray, outfile: pathlib.Path, models:List[str]) -> None:
        labels = models + ["neutral", "neutral rescaled", "slim",]
        a = self.compute_cov(a)
        b = (
            np.arange(self.size)
            / self.size
            * self.runner.sequence_length
            * self.runner.rho
        )
        marker = itertools.cycle((".", "+", "v", "^"))
        for i, model in enumerate(models):
            x = a[i]
            plt.plot(
                b, x, label=model, marker=next(marker), markersize=10, linestyle="None"
            )
        exp = np.array([self.expected_cov(r) for r in b])
        plt.plot(b, exp, marker="o", label=f"exp_hudson")
        plt.legend(loc="upper right")
        plt.savefig(outfile)
        plt.close("all")


@dataclasses.dataclass
class ExtBranchStat(SummaryStat):
    dim: int = 1
    label: str = "ext_branch_"

    def compute(self, ts: tskit.TreeSequence) -> float:
        sfs = ts.allele_frequency_spectrum(
            sample_sets=None,
            windows=None,
            mode="branch",
            span_normalise=True,
            polarised=True,
        )
        return sfs[1] / np.sum(sfs)


@dataclasses.dataclass
class DiversityStat(SummaryStat):
    dim: int = 1
    label: str = "diversity_"

    def compute(self, ts: tskit.TreeSequence) -> float:
        return ts.diversity(
            sample_sets=None,
            windows=None,
            mode="branch",
            span_normalise=True,
        )


@dataclasses.dataclass
class TajimasDStat(SummaryStat):
    dim: int = 1
    label: str = "TajimasD_"

    def compute(self, ts: tskit.TreeSequence) -> float:
        return ts.Tajimas_D(
            sample_sets=None,
            windows=None,
            mode="branch",
        )


@dataclasses.dataclass
class NumNodesStat(SummaryStat):
    dim: int = 1
    label: str = "TajimasD_"

    def compute(self, ts: tskit.TreeSequence) -> float:
        return ts.num_nodes


class SimRunner:
    def __init__(
        self, seed: int = None, num_reps: int = 100, slim_trees: pathlib.Path = None
    ):
        self.rng = np.random.default_rng(seed)
        self.num_reps = num_reps
        self.slim_trees = slim_trees

    def get_seeds(self):
        max_seed = 2**16
        return self.rng.integers(1, max_seed, size=self.num_reps)

    def _run_simulator(self, params, model="fitnessclass"):
        seeds = self.get_seeds()

        if model == "fitnessclass":

            k_map = ancestry.FitnessClassMap(
                np.array([0, params["L"]]),
                np.array(
                    [
                        0.01,
                    ]
                ),
            )
            sim = fitnessclass.Simulator(
                L=params["L"],
                r=params["r"],
                n=params["n"],
                Ne=params["Ne"],
                ploidy=2,
                K=k_map,
                U=params["U"],
                s=params["s"],
            )
            for seed in tqdm(seeds, desc="Running fitnessclass model."):
                sim.reset(seed)
                yield sim.run()
        elif model == "zeroclass":

            sim = zeroclass.Simulator(
                L=params["L"],
                r=params["r"],
                n=params["n"],
                Ne=params["Ne"],
                ploidy=2,
                U=params["U"],
                s=params["s"],
            )
            for seed in tqdm(seeds, desc="Running zeroclass model."):
                sim.reset(seed)
                yield sim.run()

        elif model == "hudson_rescaled":
            for seed in tqdm(seeds, desc="Running hudson."):
                yield msprime.sim_ancestry(
                    samples=params["n"],
                    sequence_length=params["L"],
                    recombination_rate=params["r"],
                    population_size=params["Ne"]*np.exp(- params["U"] / params["s"]),
                    random_seed=seed,
                )

        else:
            for seed in tqdm(seeds, desc="Running hudson."):
                yield msprime.sim_ancestry(
                    samples=params["n"],
                    sequence_length=params["L"],
                    recombination_rate=params["r"],
                    population_size=params["Ne"],
                    random_seed=seed,
                )

    def sample_recap_simplify(self, slim_ts, sample_size, Ne, r):
        """
        takes a ts from slim and samples, recaps, simplifies
        """
        demography = msprime.Demography.from_tree_sequence(slim_ts)
        demography.initial_size = Ne
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=msprime.IncompletePopulationMetadataWarning
            )
            recap = msprime.sim_ancestry(
                initial_state=slim_ts,
                demography=demography,
                recombination_rate=r,
                # TODO is this needed now? Shouldn't be, right?
                start_time=slim_ts.metadata["SLiM"]["generation"],
            )
        alive_inds = pyslim.individuals_alive_at(recap, 0)
        keep_indivs = np.random.choice(alive_inds, sample_size, replace=False)
        keep_nodes = []
        for i in keep_indivs:
            keep_nodes.extend(recap.individual(i).nodes)
        sts = recap.simplify(keep_nodes)
        return sts

    def _run_slim(self, slim_params):
        treefile = self.slim_trees / "temp.trees"
        slim_params["OUTFILE"] = str(treefile)
        scriptfile = self.slim_trees / "bs.slim"
        write_slim_script(scriptfile, slim_params)
        cmd = _slim_executable + [scriptfile]
        for _ in range(self.num_reps):
            subprocess.check_output(cmd)
            ts = pyslim.update(tskit.load(treefile))
            rts = self.sample_recap_simplify(
                ts, slim_params["n"], slim_params["NE"], slim_params["RHO"]
            )
            yield rts

    def _process_slim_trees(self, n):
        ts_paths = os.listdir(self.slim_trees)
        for i in range(self.num_reps):
            ts = tskit.load(self.slim_trees / ts_paths[i])
            samples = self.rng.choice(np.arange(ts.num_samples), replace=False , size=n)
            ts_simpl = ts.simplify(samples)
            yield ts_simpl

    def run_analysis(self, params, stats: List[Stat], output_dir: pathlib.Path, models: List[str]) -> None:
        results = [np.zeros((len(models) + 3, self.num_reps, stat.dim)) for stat in stats]

        # run bs simulator
        m = -1
        for m, model in enumerate(models):
            for i, ts in enumerate(self._run_simulator(params, model=model)):
                for j, stat in enumerate(stats):
                    results[j][m, i, ...] = stat.compute(ts)

        # run neutral sims
        for i, ts in enumerate(self._run_simulator(params, model="hudson")):
            for j, stat in enumerate(stats):
                results[j][m+1, i, ...] = stat.compute(ts)

        # run neutral sims
        for i, ts in enumerate(self._run_simulator(params, model="hudson_rescaled")):
            for j, stat in enumerate(stats):
                results[j][m+2, i, ...] = stat.compute(ts)

        # TODO: run bs sims (forwards in time)
        for i, ts in tqdm(enumerate(self._process_slim_trees(2 * params["n"])), total=self.num_reps):
           for j, stat in enumerate(stats):
               results[j][m+3, i, ...] = stat.compute(ts)

        for j, stat in enumerate(stats):
            output_file = output_dir / (
                stat.label
                #+ f'r{params["r"]}_U{params["U"]}_s{params["s"]}_L{params["L"]}.png'
            )
            stat.plot(results[j].squeeze(), output_file, models)


def write_slim_script(outfile, format_dict):
    slim_str = """
    // set up a simple neutral simulation
    initialize()
    {{
        initializeTreeSeq(checkCoalescence=T);
        initializeMutationRate({MU_RATE});
        initializeMutationType('m1', 0.5, 'f', {SEL_COEFF});
        // g1 genomic element type: uses m1 for all mutations
        initializeGenomicElementType('g1', m1, 1.0);
        // uniform chromosome
        initializeGenomicElement(g1, 0, {NUM_LOCI});
        // uniform recombination along the chromosome
        initializeRecombinationRate({RHO});
    }}
    // create a population
    1
    {{
        sim.addSubpop("p0", {NE});
    }}
    // run for set number of generations
    1: late()
    {{
        if (sim.treeSeqCoalesced()) {{
            sim.treeSeqOutput('{OUTFILE}');
            sim.simulationFinished();
        }}
    }}
    10000 late() {{
        sim.treeSeqOutput('{OUTFILE}');
        sim.simulationFinished();
    }}
    """
    with open(outfile, "w") as f:
        f.write(slim_str.format(**format_dict))


def get_slim_param_dict(params):
    return {
        "n": 100,
        "MU_RATE": params["U"] / params["L"] / 2,
        "SEL_COEFF": -params["s"],
        "NUM_LOCI": params["L"] - 1,
        "RHO": params["r"],
        "NE": params["Ne"],
    }


def main():
    ## SLIM PARAMS
    slim_params = {
        "L": 100_000,
        "r": 1e-8,
        "n": 100,
        "Ne": 10_000,
        "U": 2e-3,
        "s": 1e-3,
    }
    ## PARAMS
    params = {
        "L": 100_000,
        "r": 1e-8,
        "n": 4,
        "Ne": 10_000,
        "U": 2e-3,
        "s": 1e-3,
    }
    # ploidy = 2
    # slim_params = get_slim_param_dict(params)
    slim_trees = pathlib.Path("_output/trees/slim/recap")
    num_reps = 100
    SR = SimRunner(num_reps=num_reps, slim_trees=slim_trees)
    output_dir = pathlib.Path("_output/fitnessclass_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = [
        ExtBranchStat(),
        DiversityStat(),
        TajimasDStat(),
        NumNodesStat(),
    ]
    models = ["fitnessclass", "zeroclass"]
    models = ["zeroclass"]
    SR.run_analysis(params, stats, output_dir, models)


if __name__ == "__main__":
    main()
