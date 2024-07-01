import contextlib
import itertools
import msprime
import multiprocessing as mp
import numpy as np
import pathlib
import pyslim
import subprocess
import warnings
import tskit

from tqdm import tqdm
from typing import List

_slim_executable = ["./_data/slim"]


def _run_slim(scriptfile, slim_params, outdir, rep):
    treefile = outdir / f"bs_{rep}.trees"
    cmd = _slim_executable + [
        "-d",
        f"OUTFILE='{treefile}'",
        scriptfile,
    ]
    subprocess.check_output(cmd)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )
        ts = pyslim.update(tskit.load(treefile))
        rts = sample_recap_simplify(
            ts, slim_params["n"], slim_params["NE"], slim_params["RHO"]
        )
        rts.dump(outdir / f"bs_{rep}.recap")

    return 0


def sample_recap_simplify(slim_ts, sample_size, Ne, r):
    """
    takes a ts from slim and samples, recaps, simplifies
    """
    demography = msprime.Demography.from_tree_sequence(slim_ts)
    demography[0].initial_size = Ne
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=[
                msprime.IncompletePopulationMetadataWarning,
            ]
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


@contextlib.contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


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
    1 early()
    {{
        sim.addSubpop("p0", {NE});
    }}
    // run for set number of generations
    1: late()
    {{
        if (sim.treeSeqCoalesced()) {{
            sim.treeSeqOutput(OUTFILE);
            sim.simulationFinished();
        }}
    }}
    10000 late() {{
        sim.treeSeqOutput(OUTFILE);
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


class SimRunner:
    def __init__(
        self, seed: int = None, num_reps: int = 100, outdir: pathlib.Path = None
    ):
        self.rng = np.random.default_rng(seed)
        self.num_reps = num_reps
        self.outdir = outdir

    def get_seeds(self):
        max_seed = 2**16
        return self.rng.integers(1, max_seed, size=self.num_reps)

    def _run_reps(self, slim_params, num_processes=1):
        scriptfile = self.outdir / "bs.slim"
        write_slim_script(scriptfile, slim_params)
        with tqdm(total=self.num_reps, desc="Running SLiM") as pbar:
            if num_processes == 1:
                for rep, _ in enumerate(range(self.num_reps)):
                    _run_slim(scriptfile, slim_params, self.outdir, rep)
                    pbar.update(1)
            else:
                with poolcontext(processes=num_processes) as pool:
                    for simulate_window in pool.starmap(
                        _run_slim,
                        #scriptfile, slim_params, outdir, rep
                        zip(
                            itertools.repeat(scriptfile),
                            itertools.repeat(slim_params),
                            itertools.repeat(self.outdir),
                            range(self.num_reps),
                        ),
                    ):
                        pbar.update(1)


def main():
    ## PARAMS
    params = {
        "L": 100_000,
        "r": 1e-8,
        "n": 4,
        "Ne": 10_000,
        "U": 2e-3,
        "s": 1e-3,
    }
    num_processes = 10
    slim_params = get_slim_param_dict(params)
    outdir = pathlib.Path("_output/trees/slim/recap")
    num_reps = 10
    SR = SimRunner(num_reps=num_reps, outdir=outdir)
    SR._run_reps(slim_params, num_processes=num_processes)


if __name__ == "__main__":
    main()
