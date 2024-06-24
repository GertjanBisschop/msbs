import argparse
import dataclasses
import numpy as np
import msprime
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import sys

from tqdm import tqdm
from collections.abc import Iterable
from typing import Dict

matplotlib.use("Agg")

from msbs import ancestry
from msbs import fitnessclass


@dataclasses.dataclass
class TsStatRunner:

    num_reps: int
    samples: int
    r: float
    sequence_length: float
    Ne: int
    output_dir: str
    seed: int = None
    params: Dict = None

    def __post_init__(self):
        self.info_str = f"L_{self.sequence_length}_r_{self.r}"
        self.set_output_dir()
        self.models = ["localne", "hudson", "overlap"]
        self.rng = np.random.default_rng(self.seed)

    def set_output_dir(self):
        output_dir = pathlib.Path(
            self.output_dir + f"/n_{self.samples}/" + self.info_str
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def require_output_dir(self, folder_name):
        output_dir = self.output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _run_all(self, T):
        max_size = max(stat.size for stat in T)
        a = np.zeros(
            (len(self.models), len(T), self.num_reps, max_size), dtype=np.float64
        )
        for i, model in enumerate(self.models):
            self._run_single_model(model, a[i], T)
        for i, stat in enumerate(T):
            stat.plot(a[:, i, :, : stat.size])

    def _run_single_model(self, model, a, T):
        self.params["model"] = model
        if model in {"hudson", "smc"}:
            single_run = self._run_msprime()
        elif model in {"localne", "overlap"}:
            simulator = ancestry.Simulator(
                L=self.sequence_length,
                r=self.r,
                n=self.samples,
                Ne=self.Ne,
                B=self.params["b_map"],
                model=self.params["model"],
            )
            single_run = self._run_simulator(simulator)
        elif model == "fitnessclass":
            simulator = fitnessclass.Simulator(
                L=self.sequence_length,
                r=self.r,
                n=self.samples,
                Ne=self.Ne,
                K=self.params["k_map"],
                U=self.params["U"],
                s=self.params["s"],
            )
            single_run = self._run_simulator(simulator)
        else:
            raise ValueError

        for j, ts in enumerate(single_run):
            ts = ts.simplify()
            for i, stat in enumerate(T):
                a[i, j, : stat.size] = stat.compute(ts)

    def get_seeds(self):
        max_seed = 2**16
        return self.rng.integers(1, max_seed, size=self.num_reps)

    def _run_simulator(self, simulator):
        seeds = self.get_seeds()
        for seed in tqdm(seeds, desc=f"Running {self.params['model']}"):
            simulator.reset()
            yield simulator.run()

    def _run_msprime(self):
        seeds = self.get_seeds()
        model = self.params["model"]
        for seed in tqdm(seeds, desc=f"Running {model}"):
            yield msprime.sim_ancestry(
                samples=self.samples,
                recombination_rate=self.r,
                sequence_length=self.sequence_length,
                ploidy=2,
                population_size=self.Ne,
                random_seed=seed,
                model=model,
            )


@dataclasses.dataclass
class WindowStat:
    runner: TsStatRunner
    size: int = 3

    def __post_init__(self):
        self.set_output_dir()
        L = self.runner.params["L"]
        self.windows = np.array([0, 1 / 2 * L, 3 / 4 * L, L])

    @property
    def name(self):
        return self.__class__.__name__

    def set_output_dir(self):
        self.output_dir = self.runner.output_dir

    def _build_filename(self, stat_type, extension=".png"):
        return self.output_dir / (stat_type + self.name + extension)


class Diversity(WindowStat):
    def compute(self, ts):
        return ts.diversity(
            sample_sets=None,
            windows=self.windows,
            mode="branch",
            span_normalise=True,
        )

    def plot(self, a):
        mean_a = np.mean(a, axis=1)
        for i in range(mean_a.shape[0]):
            f = self._build_filename("stairs_")
            plt.stairs(mean_a[i], self.windows, label=self.runner.models[i])
        plt.legend(loc="upper right")
        plt.savefig(f, dpi=120)
        plt.close("all")


class SFS(WindowStat):
    def compute(self, ts):
        sfs = ts.allele_frequency_spectrum(
            sample_sets=None,
            windows=self.windows,
            mode="branch",
            span_normalise=True,
            polarised=True,
        )
        return sfs[:, 1] / np.sum(sfs, axis=-1)

    def plot(self, a):
        mean_a = np.mean(a, axis=1)
        for i in range(mean_a.shape[0]):
            f = self._build_filename("stairs_")
            plt.stairs(mean_a[i], self.windows, label=self.runner.models[i])
        plt.legend(loc="upper right")
        plt.savefig(f, dpi=120)
        plt.close("all")


def plot_histogram(x, x_label, filename, stat_obj):
    n, bins, patches = plt.hist(x, density=True, bins="auto")
    plt.xlabel(x_label)
    plt.ylabel("density")
    plt.savefig(filename, dpi=120)
    plt.close("all")


def plot_cdf(a, x_label, filename, stat_obj):
    for i, model in enumerate(stat_obj.models):
        x = np.sort(a[i])
        y = np.arange(len(x)) / float(len(x))
        plt.plot(x, y, label=model)
    plt.xlabel(x_label)
    plt.ylabel("cdf")
    plt.legend(loc="lower right")
    plt.savefig(filename, dpi=120)
    plt.close("all")


def plot_line(a, b, x_label, y_label, filename, stat_obj):
    for i, model in enumerate(stat_obj.models):
        x = a[i]
        plt.plot(b, x, label=model)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="upper right")
    plt.savefig(filename, dpi=120)
    plt.close("all")


def run_all(suite, output_dir, seed):
    L = 1000
    r = 7.5e-7
    num_reps = 100
    Ne = 10_000
    b_map = ancestry.BMap(
        np.array([0, 1 / 2 * L, 3 / 4 * L, L]), np.array([1.0, 0.01, 1.0])
    )
    k_map = None
    u = 2e-3
    s = 1e-3
    params = {"b_map": b_map, "L": L, "k_map": k_map, "u": u, "s": s}

    for n in [2, 4]:
        print(f"[+] Running models for n={n}")
        all_stats = []
        S = TsStatRunner(num_reps, n, r, L, Ne, output_dir, seed, params)
        for cl_name in suite:
            instance = getattr(sys.modules[__name__], cl_name)(S)
            all_stats.append(instance)
        S._run_all(all_stats)


def main():
    parser = argparse.ArgumentParser()
    choices = ["Diversity", "SFS"]

    parser.add_argument(
        "--methods",
        "-m",
        nargs="*",
        default=choices,
        choices=choices,
        help="Run all the specified methods.",
    )

    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="_output/stats_properties_ts",
        help="specify the base output directory",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="specify seed",
    )

    args = parser.parse_args()

    run_all(args.methods, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
