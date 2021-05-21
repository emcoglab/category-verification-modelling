from copy import deepcopy
from pathlib import Path
from typing import Dict

from framework.cli.job import InteractiveCombinedJob, InteractiveCombinedJobSpec

logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class Job_1(InteractiveCombinedJob):

    # max_sphere_radius (i.e. pruning distance) -> RAM/G
    def SM_RAM(self, distance: float) -> int:
        if distance <= 1:
            return 5
        elif distance <= 1.5:
            return 30
        # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        elif distance <= 1.98:
            return 55
        elif distance <= 2:
            return 60
        else:
            # Max
            return 120

    LING_RAM: Dict[str, Dict[int, int]] = {
        "pmi_ngram": {
            1_000: 2,
            3_000: 3,
            10_000: 7,
            30_000: 11,
            40_000: 15,
            60_000: 20,
        },
        "ppmi_ngram": {
            1_000: 2,
            3_000: 3,
            10_000: 5,
            30_000: 7,
            40_000: 9,
            60_000: 11,
        }
    }

    def __init__(self, spec: InteractiveCombinedJobSpec):
        super().__init__(
            script_number="1",
            script_name="1_modelling.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, InteractiveCombinedJobSpec)
        return self.SM_RAM(self.spec.sensorimotor_spec.max_radius) \
               + self.LING_RAM[self.spec.linguistic_spec.model_name][self.spec.linguistic_spec.n_words]


if __name__ == '__main__':

    jobs = []
    s: InteractiveCombinedJobSpec
    for s in InteractiveCombinedJobSpec.load_multiple(
            Path(Path(__file__).parent, "job_specifications/2021-05-21 good example model.yaml")):
        jobs.append(Job_1(s))

    for job in jobs:
        job.run_locally(extra_arguments=["--filter_events accessible_set"])

    print(f"Submitted {len(jobs)} jobs.")
