import abc
import click
import dataclasses
import itertools
import matplotlib.pyplot as plt
import msprime
import numpy as np
import os
import pathlib
import pyslim
import random
import subprocess
import warnings
import tskit

from tqdm import tqdm
from typing import List, Dict

from msbs import ancestry
from msbs import fitnessclass
from msbs import zeroclass
from msbs import nett
from msbs import utils

_slim_executable = ["./_data/slim"]


@dataclasses.dataclass
class Stat:
    dim: int
    label: str

    @abc.abstractmethod
    def compute(self, ts: tskit.TreeSequence, **kwargs: float) -> float:
        return

    @abc.abstractmethod
    def plot(data: np.ndarray, outfile: pathlib.Path, models: List[str]) -> None:
        return


@dataclasses.dataclass
class SummaryStat(Stat):
    norm: bool = False

    @abc.abstractmethod
    def compute(self, ts: tskit.TreeSequence, **kwargs: float) -> float:
        return

    def plot(self, data: np.ndarray, outfile: pathlib.Path, models: List[str]) -> None:
        if self.norm:
            data /= np.mean(data[-3])
        plt.violinplot([z[~np.isnan(z)] for z in data], showmeans=True)
        plt.xticks(
            [y + 1 for y in range(data.shape[0])],
            labels=models
            + [
                "neutral",
                "neutral rescaled",
                "slim",
            ],
        )
        plt.title(self.label)
        plt.savefig(outfile, dpi=120)
        plt.close("all")


@dataclasses.dataclass
class CovStat(Stat):
    dim: int = 20
    label: str = "CovDecay_"
    r: float = 0.0
    L: float = 100_000
    Ne: float = 10_000

    def __post_init__(self):
        self.max_length = 10 / (self.r * self.Ne)

    def compute(self, ts: tskit.TreeSequence, **kwargs: float) -> float:
        # requires many observations of t_i and t_j at given distances
        # t_i can stay same fixed value
        num_points = self.dim
        result = np.zeros(num_points, dtype=np.float64)
        points = np.arange(num_points + 1) / (num_points + 1) * self.max_length
        u, v = random.sample(range(ts.num_samples), 2)

        # collect info on pairwaise coalescence times
        tree = ts.at(points[0])
        coal_time_uv = tree.tmrca(u, v)
        for i in range(num_points):
            tree = ts.at(points[i + 1])
            result[i] = tree.tmrca(u, v) == coal_time_uv

        return result

    def group(self, all_reps, num_models):
        # shape a: num_models, num_reps, num_points
        data = np.ma.array(all_reps, mask=np.isnan(all_reps))
        num_reps = all_reps.shape[1]
        num_points = self.dim
        result = np.zeros((num_models, num_points), dtype=np.float64)
        for i in range(num_models):
            for j in range(num_points):
                result[i, j] = np.array(np.sum(data[i, :, j]).data) / num_reps

        return result

    def expected_cov(self, r):
        return (r + 18) / (r**2 + 13 * r + 18)

    def plot(self, data: np.ndarray, outfile: pathlib.Path, models: List[str]) -> None:
        labels = models + [
            "neutral",
            "neutral rescaled",
            "slim",
        ]
        # group observations: shape: (num_models, num_reps, num_points)
        a = self.group(data, len(labels))
        b = (
            np.arange(self.dim + 1)
            / (self.dim + 1)
            * self.max_length
            * self.r
            * 4
            * self.Ne
        )
        marker = itertools.cycle((".", "+", "v", "^"))
        for i, model in enumerate(labels):
            x = a[i]
            plt.plot(
                b[1:],
                x,
                label=model,
                marker=next(marker),
                markersize=10,
                linestyle="None",
            )
        exp = np.array([self.expected_cov(r) for r in b])
        plt.plot(b, exp, marker=None, label=f"exp_hudson")
        plt.legend(loc="upper right")
        plt.title(self.label)
        plt.xlabel("rho")
        plt.ylabel("covariance")
        plt.savefig(outfile)
        plt.close("all")


@dataclasses.dataclass
class SFSStat(Stat):
    dim: int
    label: str = "SFS_"

    def compute(self, ts: tskit.TreeSequence, q: float = 1.0) -> np.ndarray:
        afs = ts.allele_frequency_spectrum(
            polarised=True, mode="branch", span_normalise=True
        )[1:-1]
        return afs * q

    def group(self, all_reps):
        data = np.ma.array(all_reps, mask=np.isnan(all_reps))
        mean_over_reps = np.mean(data, axis=1).data
        return mean_over_reps

    def plot(self, data: np.ndarray, outfile: pathlib.Path, models: List[str]) -> None:
        self._plot_absolute(data, outfile, models)
        self._plot_relative_error(data, outfile, models)

    def _plot_relative_error(
        self, data: np.ndarray, outfile: pathlib.Path, models: List[str]
    ) -> None:
        labels = models + [
            "neutral",
            "neutral rescaled",
            "slim",
        ]
        # group observations: shape: (num_models, num_reps, num_points)
        a = self.group(data)
        a_rel = np.abs(a - a[-1])
        if self.dim <= 10:
            x_axis = np.arange(1, a.shape[-1] + 1)
            p = len(x_axis)
            p1 = p * 2.5
            p2 = p * 1.5
            fig = plt.figure(figsize=(p1, p2))
            ax = fig.gca()
            num_bars = len(labels) - 1
            width = 1 / (num_bars)
            half_width = 1 / 2 * width
            half_num_bars = np.floor(num_bars / 2)

            for i in range(num_bars):
                j = i - half_num_bars
                counts = a_rel[i]
                ax.bar(
                    x_axis + j * half_width,
                    counts,
                    label=labels[i],
                    width=half_width,
                    align="center",
                )
            ax.xaxis.set_tick_params(which="major", labelsize=p2)

        else:
            # continuous plot
            # only plot first q alleles
            q = self.dim // 5
            x_axis = np.arange(1, q)
            fig = plt.figure()
            ax = fig.gca()
            for i in range(len(labels) - 1):
                ax.plot(x_axis, a_rel[i, : q - 1], label=labels[i], marker=".")

        ax.legend(loc="upper right")
        plt.title(self.label)
        parts = list(outfile.parts)
        temp = parts[-1].split("_")
        temp[0] = "SFSRel"
        parts[-1] = "_".join(temp)
        fig.savefig(pathlib.Path(*parts))
        plt.close("all")

    def _plot_absolute(
        self, data: np.ndarray, outfile: pathlib.Path, models: List[str]
    ) -> None:
        labels = models + [
            "neutral",
            "neutral rescaled",
            "slim",
        ]
        # group observations: shape: (num_models, num_reps, num_points)
        a = self.group(data)
        if self.dim <= 10:
            x_axis = np.arange(1, a.shape[-1] + 1)
            p = len(x_axis)
            p1 = p * 2.5
            p2 = p * 1.5
            fig = plt.figure(figsize=(p1, p2))
            ax = fig.gca()
            num_bars = len(labels)
            width = 1 / (num_bars)
            half_width = 1 / 2 * width
            half_num_bars = np.floor(num_bars / 2)

            for i in range(num_bars):
                j = i - half_num_bars
                counts = a[i]
                ax.bar(
                    x_axis + j * half_width,
                    counts,
                    label=labels[i],
                    width=half_width,
                    align="center",
                )
            ax.xaxis.set_tick_params(which="major", labelsize=p2)

        else:
            # continuous plot
            # only plot first q alleles
            q = self.dim // 5
            d = np.log(a)
            x_axis = np.arange(1, q)
            fig = plt.figure()
            ax = fig.gca()
            for i in range(len(labels) - 1):
                ax.plot(x_axis, d[i, : q - 1], label=labels[i], marker=".")

            ax.plot(x_axis, d[-1, : q - 1], label=labels[-1], marker="o")

        ax.legend(loc="upper right")
        plt.title(self.label)
        fig.savefig(outfile)
        plt.close("all")


@dataclasses.dataclass
class ExtBranchStat(SummaryStat):
    dim: int = 1
    label: str = "ext_branch_"
    norm: bool = True

    def compute(self, ts: tskit.TreeSequence, **kwargs: float) -> float:
        sfs = ts.allele_frequency_spectrum(
            sample_sets=None,
            windows=None,
            mode="branch",
            span_normalise=True,
            polarised=True,
        )
        return sfs[1] / np.sum(sfs)


@dataclasses.dataclass
class OldestRootStat(SummaryStat):
    dim: int = 1
    label: str = "oldest_root_"
    norm: bool = True

    def compute(self, ts: tskit.TreeSequence, q: float = 1.0) -> float:
        return ts.max_root_time * q


@dataclasses.dataclass
class DiversityStat(SummaryStat):
    dim: int = 1
    label: str = "diversity_"
    norm: bool = True

    def compute(self, ts: tskit.TreeSequence, q: float = 1.0) -> float:
        return ts.diversity(
            sample_sets=None,
            windows=None,
            mode="branch",
            span_normalise=True,
        ) * q


@dataclasses.dataclass
class TajimasDStat(SummaryStat):
    dim: int = 1
    label: str = "TajimasD_"

    def compute(self, ts: tskit.TreeSequence, **kwargs: float) -> float:
        return ts.Tajimas_D(
            sample_sets=None,
            windows=None,
            mode="branch",
        )


@dataclasses.dataclass
class NumNodesStat(SummaryStat):
    dim: int = 1
    label: str = "NumNodes_"
    norm: bool = True

    def compute(self, ts: tskit.TreeSequence, **kwargs: float) -> float:
        return ts.num_nodes


@dataclasses.dataclass
class NumTreesStat(SummaryStat):
    dim: int = 1
    label: str = "NumTrees_"
    norm: bool = True

    def compute(self, ts: tskit.TreeSequence, **kwargs: float) -> float:
        return ts.num_trees


@dataclasses.dataclass
class MidTreeTBL(SummaryStat):
    dim: int = 1
    label: str = "MidTreeTBL_"
    norm: bool = True
    mid: int = 0

    def compute(self, ts: tskit.TreeSequence, q: float = 1.0) -> float:
        return ts.at(self.mid).total_branch_length * q


@dataclasses.dataclass
class FirstTreeTBL(SummaryStat):
    dim: int = 1
    label: str = "FirstTreeTBL_"
    norm: bool = True

    def compute(self, ts: tskit.TreeSequence, q: float = 1.0) -> float:
        return ts.first().total_branch_length * q


@dataclasses.dataclass
class MidTreeB2(SummaryStat):
    dim: int = 1
    label: str = "MidTreeB2_"
    mid: int = 0

    def compute(self, ts: tskit.TreeSequence, **kwargs: float) -> float:
        return ts.at(self.mid).b2_index(base=2)


@dataclasses.dataclass
class MidTreeColless(SummaryStat):
    dim: int = 1
    label: str = "MidTreeColless_"
    mid: int = 0

    def compute(self, ts: tskit.TreeSequence, **kwargs: float) -> float:
        return ts.at(self.mid).colless_index()


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

    def _run_simulator(self, params, n, model="fitnessclass"):
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
                n=n,
                Ne=params["Ne"],
                ploidy=2,
                K=k_map,
                U=params["U"],
                s=params["s"],
            )
            for seed in tqdm(seeds, desc="Running fitnessclass model"):
                sim.reset(seed)
                yield sim.run()
        elif model == "zeroclass":

            sim = zeroclass.ZeroClassSimulator(
                L=params["L"],
                r=params["r"],
                n=n,
                Ne=params["Ne"],
                ploidy=2,
                U=params["U"],
                s=params["s"],
            )
            for seed in tqdm(seeds, desc="Running zeroclass model"):
                sim.reset(seed)
                yield sim.run(ca_events=True, end_time=None)

        elif model == "zeroclassemulator":

            sim = zeroclass.ZeroClassEmulator(
                L=params["L"],
                r=params["r"],
                n=n,
                Ne=params["Ne"],
                ploidy=2,
                U=params["U"],
                s=params["s"],
            )
            for seed in tqdm(seeds, desc="Running zeroclass emulation model"):
                sim.reset(seed)
                yield sim.run()

        elif model == "hudson_rescaled":
            rescale = np.exp(-params["U"] / params["s"])
            if rescale < 1:
                R = params["r"] * params["L"]
                rescale = np.exp(-params["U"] / (params["s"] + R / 2))
            for seed in tqdm(seeds, desc="Running hudson rescaled"):
                yield msprime.sim_ancestry(
                    samples=n,
                    sequence_length=params["L"],
                    recombination_rate=params["r"],
                    population_size=params["Ne"] * rescale,
                    random_seed=seed,
                )

        elif model == "stepwise":
            demography = utils.stepwise_factory(
                params["Ne"],
                np.arange(1, 11) * 1000,
                np.array(
                    [10_000, 9_325, 8820, 8173, 7630, 6840, 6807, 5795, 5790, 5375]
                ),
            )
            sim = nett.StepWiseSimulator(
                L=params["L"],
                r=params["r"],
                n=n,
                Ne=params["Ne"],
            )

            for seed in tqdm(seeds, desc="Running stepwise."):
                yield sim.run(demography, seed=seed)

        else:
            for seed in tqdm(seeds, desc="Running hudson"):
                yield msprime.sim_ancestry(
                    samples=n,
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

    def _run_slim(self, slim_params, n):
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

    def _process_slim_trees(self, n, L):
        ts_paths = os.listdir(self.slim_trees)
        num_reps = min(self.num_reps, len(ts_paths))
        if num_reps < self.num_reps:
            print("[X] Number of replicates requested is larger than the number of available SLiM trees.")
        for i in range(num_reps):
            ts = tskit.load(self.slim_trees / ts_paths[i])
            if ts.sequence_length > L + 1:
                mid = ts.sequence_length // 2
                interval = [(mid - L // 2, mid + L // 2)]
                ts = ts.keep_intervals(interval).ltrim().rtrim()
            samples = self.rng.choice(np.arange(ts.num_samples), replace=False, size=n)
            ts_simpl = ts.simplify(samples)
            yield ts_simpl

    def run_analysis(
        self,
        params: Dict,
        n: int,
        stats: List[Stat],
        output_dir: pathlib.Path,
        models: List[str],
    ) -> None:
        results = [
            np.full((len(models) + 3, self.num_reps, stat.dim), dtype=np.float64, fill_value=np.nan) for stat in stats
        ]

        # run bs simulator
        m = -1
        for m, model in enumerate(models):
            for i, ts in enumerate(self._run_simulator(params, n, model=model)):
                for j, stat in enumerate(stats):
                    results[j][m, i, ...] = stat.compute(ts)

        # run neutral sims
        for i, ts in enumerate(self._run_simulator(params, n, model="hudson")):
            for j, stat in enumerate(stats):
                results[j][m + 1, i, ...] = stat.compute(ts)

        # run neutral sims
        for i, ts in enumerate(self._run_simulator(params, n, model="hudson_rescaled")):
            for j, stat in enumerate(stats):
                results[j][m + 2, i, ...] = stat.compute(ts)

        # analyse forwards in time sims
        for i, ts in tqdm(
            enumerate(self._process_slim_trees(2 * n, params["L"])),
            total=self.num_reps,
            desc="SLiM trees",
        ):
            for j, stat in enumerate(stats):
                results[j][m + 3, i, ...] = stat.compute(ts, q=params["q"])

        for j, stat in enumerate(stats):
            output_file = output_dir / (stat.label + f"n{n}.png")
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


@click.command()
@click.option("--scenario", default="simple")
@click.option("--slim", default=None)
@click.option("--n", default=5)
@click.option("--reps", default=100)
def compare(scenario, slim, n, reps):
    possible_scenarios = {"human", "dros", "human_weak", "human_strong", "simple"}
    if not scenario in possible_scenarios:
        click.echo("Scenario not implemented.")
        raise SystemExit(1)
    if slim is None:
        slim_trees = pathlib.Path(f"_output/trees/slim/{scenario}")
    else:
        slim_trees = pathlib.Path(slim)
    if not slim_trees.exists():
        click.echo("Slim directory does not exist.")
        raise SystemExit(1)

    ## PARAMS
    temp_L = 1_000_000
    params_scenarios = {
        "simple": {  # U/s = 1, Ns*e**(-U/s) = 3.67, Ns = 10
            "L": 100_000,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 1e-3,
            "s": 1e-3,
            "q": 1, # scaling factor
        },
        "human": {  # U/s = 18, Ns*e**(-U/s) = 3.8e-7, Ns = 25
            # "L": 130_000_000,
            "L": temp_L,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 0.045 / 130_000_000 * temp_L,
            "s": 2.5e-3,
            "q": 1, # scaling factor
        },
        "human_weak": {  # U/s = 180, Ns*e**(-U/s) = 1.6e-70, Ns = 2.5
            "L": 130_000_000,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 0.045,
            "s": 2.5e-4,
            "q": 1, # scaling factor
        },
        "human_strong": {  # U/s = 1.8, Ns*e**(-U/s) = 41, Ns = 250
            "L": 130_000_000,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 0.045,
            "s": 2.5e-2,
            "q": 1, # scaling factor
        },
        "dros": {  # U/s = 0.4, Ns = 2.5e2
            "L": 24_000,
            "r": 1e-8,
            "Ne": 1_000_000,
            "U": 0.1 / 1000,
            "s": 2.5e-4,
            "q": 20, # scaling factor
        },
    }
    params = params_scenarios[scenario]
    # ploidy = 2
    # num_reps = 1000
    SR = SimRunner(num_reps=reps, slim_trees=slim_trees)
    output_dir = pathlib.Path(f"_output/compare/{scenario}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = [
        ExtBranchStat(),
        DiversityStat(),
        TajimasDStat(),
        NumNodesStat(),
        NumTreesStat(),
        FirstTreeTBL(),
        MidTreeTBL(mid=params["L"] // 2),
        MidTreeB2(mid=params["L"] // 2),
        OldestRootStat(),
        CovStat(r=params["r"], L=params["L"], Ne=params["Ne"]),
        SFSStat(dim=n * 2 - 1),
    ]
    # models = ["fitnessclass", "zeroclass"]
    models = ["zeroclass", "zeroclassemulator"]
    # models = []
    SR.run_analysis(params, n, stats, output_dir, models)


if __name__ == "__main__":
    compare()
