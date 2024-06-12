import argparse
import dataclasses
import pathlib
import numpy as np
import sys
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Dict

from msbs import zeng


@dataclasses.dataclass
class SimStat:
    output_dir: str
    seed: int = None

    def __post_init__(self):
        self.info_str = self.__class__.__name__
        self.set_output_dir()
        self.rng = np.random.default_rng(self.seed)

    def set_output_dir(self):
        output_dir = pathlib.Path(self.output_dir + "/" + self.info_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def require_output_dir(self, folder_name):
        output_dir = self.output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_seeds(self, num_reps):
        max_seed = 2**16
        return self.rng.integers(1, max_seed, size=num_reps)


@dataclasses.dataclass
class NumEventsStat(SimStat):
    def _run(self):
        num_reps = 100
        s = 0.01
        U_range = np.linspace(0.1, 0.5, num=5)
        r = 0.75e-4
        samples = 4
        print(U_range / s)
        A = self._sim(num_reps, U_range, r, s, samples)
        name = f"_s{s}_r{r}_n{samples}"
        self._plot(A, U_range / s, name)

    def _sim(self, num_reps, u_range, r, s, samples):
        sequence_length = 100
        Ne = 10_000
        result = np.zeros((num_reps, u_range.size, 3))

        seeds = self.get_seeds(num_reps)
        for i, seed in tqdm(
            enumerate(seeds),
            desc=f"Running NumEventsStat Zeng",
            total=num_reps,
        ):
            for j, U in enumerate(u_range):
                sim = zeng.ZSimulator(
                    L=sequence_length,
                    r=r,
                    n=samples,
                    Ne=Ne,
                    seed=seed,
                    U=U,
                    s=s,
                )
                sim.run()
                result[i, j, 0] = sim.num_ca_events
                result[i, j, 1] = sim.num_re_events
                result[i, j, 2] = sim.num_mu_events

        return result

    def _plot(self, a, labels, name=""):
        labels = np.round(labels, 2)
        mean_a = np.mean(a, axis=0)
        ticks = np.arange(mean_a.shape[1])
        for i in range(mean_a.shape[0]):
            plt.plot(ticks, mean_a[i], marker="o", label=labels[i])
        plt.xticks(ticks=ticks, labels=["ca", "re", "mut"])
        plt.legend(title="U")
        plt.savefig(self.output_dir / f"hist_num_events{name}.png", dpi=120)
        plt.close("all")


def run_all(suite, output_dir, seed):
    for cl_name in suite:
        instance_class = getattr(sys.modules[__name__], cl_name)
        instance = instance_class(output_dir, seed)
        if issubclass(instance_class, SimStat):
            instance._run()


def main():
    parser = argparse.ArgumentParser()
    choices = [
        "NumEventsStat",
    ]

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
        default="_output/sim_properties",
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
