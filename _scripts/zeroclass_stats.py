import abc
import click
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import statsmodels.api as sm
import tskit

from tqdm import tqdm
from typing import List, Dict

from msbs import zeroclass


@dataclasses.dataclass
class Stat:
    dim: int
    label: str
    tree_seq: bool

    @abc.abstractmethod
    def compute(self, ts: tskit.TreeSequence) -> float:
        return

    @abc.abstractmethod
    def plot(self, data: np.ndarray, outfile: pathlib.Path) -> None:
        return


@dataclasses.dataclass
class OneDimStat:
    dim: int = 1
    label: str = None
    tree_seq: bool = True

    @abc.abstractmethod
    def plot(self, data: np.ndarray, outfile: pathlib.Path) -> None:
        sm.qqplot_2samples(
            data[0], data[1], xlabel="zeroclass", ylabel="SLiM", line="45"
        )
        plt.savefig(outfile)
        plt.close("all")


@dataclasses.dataclass
class OldestMutStat(OneDimStat):
    label: str = "OldestMut_"

    def compute(self, ts: tskit.TreeSequence) -> float:
        if ts.num_mutations == 0:
            return ts.max_root_time
        else:
            return np.max(ts.tables.mutations.time)


@dataclasses.dataclass
class NumRootsStat(OneDimStat):
    label: str = "NumRoots_"

    def compute(self, ts: tskit.TreeSequence) -> float:
        all_roots = set()
        for tree in ts.trees():
            all_roots = all_roots.union(set(tree.roots))

        return len(all_roots)


@dataclasses.dataclass
class NumTreesStat(OneDimStat):
    label: str = "NumTrees_"

    def compute(self, ts: tskit.TreeSequence) -> float:
        return ts.num_trees


@dataclasses.dataclass
class NumNodesStat(OneDimStat):
    label: str = "NumNodes_"

    def compute(self, ts: tskit.TreeSequence) -> float:
        return ts.num_nodes


@dataclasses.dataclass
class SFSStat(Stat):
    dim: int
    label: str = "SFS_"
    tree_seq: bool = True

    def compute(self, ts: tskit.TreeSequence, q: float = 1.0) -> np.ndarray:
        afs = ts.allele_frequency_spectrum(
            polarised=True, mode="branch", span_normalise=True
        )[1:-1]
        return afs * q

    def group(self, all_reps):
        data = np.ma.array(all_reps, mask=np.isnan(all_reps))
        mean_over_reps = np.mean(data, axis=1).data
        return mean_over_reps

    def plot(self, data: np.ndarray, outfile: pathlib.Path) -> None:
        self._plot_absolute(data, outfile)

    def _plot_absolute(self, data: np.ndarray, outfile: pathlib.Path) -> None:
        labels = ["mutliclass", "SLiM"]
        # group observations: shape: (2, num_reps, num_points)
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
            ax.set_ylabel("branch length")
            ax.set_xlabel("iton")
            ax.xaxis.set_tick_params(which="major", labelsize=p2)

        else:
            # continuous plot
            # only plot first q alleles
            q = self.dim // 5
            x_axis = np.arange(1, q)
            fig = plt.figure()
            ax = fig.gca()
            for i in range(len(labels) - 1):
                ax.plot(x_axis, a[i, : q - 1], label=labels[i], marker=".")

            ax.plot(x_axis, a[-1, : q - 1], label=labels[-1], marker="o")
            ax.set_ylabel("branch length")
            ax.set_xlabel("iton")

        ax.legend(loc="upper right")
        plt.title(self.label)
        fig.savefig(outfile)
        plt.close("all")


@dataclasses.dataclass
class SimStat:
    dim: int = 1
    label: str = None
    tree_seq: bool = False

    @abc.abstractmethod
    def plot(self, data: np.ndarray, outfile: pathlib.Path) -> None:
        plt.hist(data[0], density=True)
        plt.title(self.label)
        plt.savefig(outfile)
        plt.close("all")


@dataclasses.dataclass
class NumCoalEventsStat(SimStat):
    dim: int = 1
    label: str = "NumCoalEvents_"
    tree_seq: bool = False

    def compute(self, sim: zeroclass.Simulator) -> float:
        if isinstance(sim, tskit.TreeSequence):
            return sim.num_nodes
        return np.sum(sim.num_coal_events)


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

    def _run_simulator(self, params, n, add_neutral):
        seeds = self.get_seeds()
        sim = zeroclass.MultiClassSimulator(
            L=params["L"],
            r=params["r"],
            n=n,
            Ne=params["Ne"],
            ploidy=2,
            U=params["U"],
            s=params["s"],
            num_populations=5,
        )

        for seed in tqdm(seeds, desc="Running zeroclass model."):
            sim.reset(seed)
            end_time = None
            ts = sim._initial_setup(ca_events=True, end_time=end_time)
            end_time = ts.max_root_time
            if add_neutral:
                tsplus = sim._complete(ts, end_time=end_time)
            else:
                tsplus = ts
            yield (tsplus, sim)

    def _process_slim_trees(self, n, L, end_times, end_idxs):
        ts_paths = os.listdir(self.slim_trees)
        num_reps = min(self.num_reps, len(ts_paths))
        if num_reps < self.num_reps:
            print(
                "[X] Number of replicates requested is larger than the number of available SLiM trees."
            )
        for i in range(num_reps):
            end_time = end_times[end_idxs[i]]
            ts = tskit.load(self.slim_trees / ts_paths[i])
            if ts.sequence_length > L + 1:
                mid = ts.sequence_length // 2
                interval = [(mid - L // 2, mid + L // 2)]
                ts = ts.keep_intervals(interval).ltrim().rtrim()
            assert ts.sequence_length == L - 1
            if end_time is not None:
                ts = ts.decapitate(end_time)
                assert ts.max_root_time == end_time
            samples = self.rng.choice(np.arange(ts.num_samples), replace=False, size=n)
            ts_simpl = ts.simplify(samples)
            yield ts_simpl

    def run_analysis(
        self,
        params: Dict,
        n: int,
        stats: List[Stat],
        output_dir: pathlib.Path,
        rescale: float,
        add_neutral: bool,
    ) -> None:
        results = [np.zeros((2, self.num_reps, stat.dim)) for stat in stats]
        max_root_time = np.zeros(self.num_reps)

        # run bs simulator
        for i, (ts, sim) in enumerate(self._run_simulator(params, n, add_neutral)):
            for j, stat in enumerate(stats):
                res = ts if stat.tree_seq else sim
                results[j][0, i, ...] = stat.compute(res)
                # always extract max root time
                max_root_time[i] = ts.max_root_time

        # analyse forwards in time sims
        max_root_idxs = self.rng.integers(
            0, max_root_time.size, size=max_root_time.size
        )
        for i, ts in tqdm(
            enumerate(
                self._process_slim_trees(
                    2 * n, params["L"], max_root_time, max_root_idxs
                )
            ),
            total=self.num_reps,
            desc="SLiM trees",
        ):
            for j, stat in enumerate(stats):
                results[j][1, i, ...] = stat.compute(ts)

        for j, stat in enumerate(stats):
            neutr = "T" if add_neutral else "F"
            output_file = output_dir / (
                stat.label + f"n{n}_scale{int(rescale)}_neutr{neutr}.png"
            )
            stat.plot(results[j].squeeze(), output_file)


@click.command()
@click.option("--scenario", default="simple")
@click.option("--n", default=5)
@click.option("--reps", default=100)
@click.option("--scale", default=1)
@click.option("--neutral/--no-neutral", default=False)
def evaluate(scenario, n, reps, scale, neutral):
    possible_scenarios = {"human", "dros", "human_weak", "human_strong", "simple"}
    if not scenario in possible_scenarios:
        click.echo("Scenario not implemented.")
        raise SystemExit(1)

    ## PARAMS
    resize_factor = scale
    params_scenarios = {
        "simple": {  # U/s = 1, Ns*e**(-U/s) = 3.67
            "L": 100_000 // resize_factor,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 1e-3 / resize_factor,
            "s": 1e-3,
        },
        "human": {  # U/s = 37, Ns*e**(-U/s) = 3.8e-7
            "L": 130_000_000 // resize_factor,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 0.045 / resize_factor,
            "s": 1.25e-3,
        },
        "human_weak": {  # U/s = 370, Ns*e**(-U/s) = 1.6e-70
            "L": 130_000_000 // resize_factor,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 0.045 / resize_factor,
            "s": 1.25e-4,
        },
        "human_strong": {  # U/s = 3.7, Ns*e**(-U/s) = 41
            "L": 130_000_000 // resize_factor,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 0.045 / resize_factor,
            "s": 1.25e-2,
        },
        "dros": {  # U/s = 50, Ns*e**(-U/s) = 3.8e-19
            "L": 24_000_000 // resize_factor,
            "r": 1e-8,
            "Ne": 1_000_000,
            "U": 0.1 / resize_factor,
            "s": 2e-3,
        },
    }
    params = params_scenarios[scenario]
    slim_trees = pathlib.Path(f"_output/trees/slim/{scenario}")
    SR = SimRunner(None, reps, slim_trees=slim_trees)
    output_dir = pathlib.Path(f"_output/zeroclass_stats/{scenario}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = [
        NumRootsStat(),
        NumTreesStat(),
        NumCoalEventsStat(),
        NumNodesStat(),
        SFSStat(dim=n * 2 - 1),
        OldestMutStat(),
    ]
    SR.run_analysis(params, n, stats, output_dir, resize_factor, neutral)


if __name__ == "__main__":
    evaluate()
