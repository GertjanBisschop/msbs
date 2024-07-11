import abc
import click
import dataclasses
import numpy as np
import pathlib
import matplotlib.pyplot as plt
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
        plt.hist(data, density=True)
        plt.title(self.label)
        plt.savefig(outfile)
        plt.close("all")


@dataclasses.dataclass
class RootTimeStat(OneDimStat):
    label: str = "RootTime_"

    def compute(self, ts: tskit.TreeSequence) -> float:
        return ts.max_root_time


@dataclasses.dataclass
class NumRootsStat(OneDimStat):
    label: str = "NumRoots_"

    def compute(self, ts: tskit.TreeSequence) -> float:
        all_roots = set()
        for tree in ts.trees():
            all_roots = all_roots.union(set(tree.roots))

        return len(all_roots)


@dataclasses.dataclass
class SimStat:
    dim: int = 1
    label: str = None
    tree_seq: bool = False

    @abc.abstractmethod
    def plot(self, data: np.ndarray, outfile: pathlib.Path) -> None:
        plt.hist(data, density=True)
        plt.title(self.label)
        plt.savefig(outfile)
        plt.close("all")


@dataclasses.dataclass
class NumMutsStat(SimStat):
    dim: int = 1
    label: str = "NumMuts_"
    tree_seq: bool = False

    def compute(self, sim: zeroclass.Simulator) -> float:
        return


class SimRunner:
    def __init__(self, seed: int = None, num_reps: int = 100):
        self.rng = np.random.default_rng(seed)
        self.num_reps = num_reps

    def get_seeds(self):
        max_seed = 2**16
        return self.rng.integers(1, max_seed, size=self.num_reps)

    def _run_simulator(self, params, n):
        seeds = self.get_seeds()
        sim = zeroclass.ZeroClassSimulator(
            L=params["L"],
            r=params["r"],
            n=n,
            Ne=params["Ne"],
            ploidy=2,
            U=params["U"],
            s=params["s"],
            bounded=True,
        )

        for seed in tqdm(seeds, desc="Running zeroclass model."):
            sim.reset(seed)
            ts = sim._initial_setup(ca_events=True)
            yield (ts, sim)

    def run_analysis(
        self,
        params: Dict,
        n: int,
        stats: List[Stat],
        output_dir: pathlib.Path,
    ) -> None:
        results = [np.zeros((self.num_reps, stat.dim)) for stat in stats]

        # run bs simulator
        for i, (ts, sim) in enumerate(self._run_simulator(params, n)):
            for j, stat in enumerate(stats):
                res = ts if stat.tree_seq else sim
                results[j][i, ...] = stat.compute(res)

        for j, stat in enumerate(stats):
            output_file = output_dir / (stat.label + f"n{n}.png")
            stat.plot(results[j].squeeze(), output_file)


@click.command()
@click.option("--scenario", default="simple")
@click.option("--n", default=5)
@click.option("--reps", default=100)
def evaluate(scenario, n, reps):
    possible_scenarios = {"human", "dros", "human_weak", "human_strong", "simple"}
    if not scenario in possible_scenarios:
        click.echo("Scenario not implemented.")
        raise SystemExit(1)

    ## PARAMS
    temp_L = 1_000_000
    params_scenarios = {
        "simple": {  # U/s = 1, Ns*e**(-U/s) = 3.67
            "L": 100_000,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 1e-3,
            "s": 1e-3,
        },
        "human": {  # U/s = 18, Ns*e**(-U/s) = 3.8e-7
            # "L": 130_000_000,
            "L": temp_L,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 0.045 / 130_000_000 * temp_L,
            "s": 2.5e-3,
        },
        "human_weak": {  # U/s = 180, Ns*e**(-U/s) = 1.6e-70
            "L": 130_000_000,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 0.045,
            "s": 2.5e-4,
        },
        "human_strong": {  # U/s = 1.8, Ns*e**(-U/s) = 41
            "L": 130_000_000,
            "r": 1e-8,
            "Ne": 10_000,
            "U": 0.045,
            "s": 2.5e-2,
        },
        "dros": {  # U/s = 50, Ns*e**(-U/s) = 3.8e-19
            "L": 24_000_000,
            "r": 1e-8,
            "Ne": 1_000_000,
            "U": 0.1,
            "s": 2e-3,
        },
    }
    params = params_scenarios[scenario]
    SR = SimRunner(None, reps)
    output_dir = pathlib.Path(f"_output/zeroclass_stats/{scenario}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = [
        RootTimeStat(),
        NumRootsStat(),
    ]
    SR.run_analysis(params, n, stats, output_dir)


if __name__ == "__main__":
    evaluate()
