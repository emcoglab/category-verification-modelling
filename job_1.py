from pathlib import Path
from typing import Dict

from framework.cli.job import CategoryVerificationJob, CategoryVerificationJobSpec


class Job_1(CategoryVerificationJob):

    # max_sphere_radius (i.e. pruning distance) -> RAM/G
    def SM_RAM(self, distance: float) -> int:
        if distance <= 1:
            return 5
        elif distance <= 1.5:
            return 20
        # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        elif distance <= 1.98:
            return 45
        elif distance <= 2:
            return 50
        else:
            # Max
            return 100

    LING_RAM: Dict[str, Dict[int, int]] = {
        "ppmi_ngram": {
            1_000: 2,
            3_000: 3,
            10_000: 5,
            30_000: 7,
            40_000: 8,
            60_000: 10,
        }
    }

    def __init__(self, spec: CategoryVerificationJobSpec):
        super().__init__(
            script_number="1",
            script_name="1_modelling.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, CategoryVerificationJobSpec)
        return (self.SM_RAM(self.spec.sensorimotor_spec.max_radius)
                + self.LING_RAM[self.spec.linguistic_spec.model_name][self.spec.linguistic_spec.n_words])


if __name__ == '__main__':

    ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    jobs = []
    s: CategoryVerificationJobSpec
    for s in CategoryVerificationJobSpec.load_multiple(Path(Path(__file__).parent,
                                                            "job_specifications/2022-08-20 good roc model with cut connections.yaml")):
        jobs.append(Job_1(s))

    job_count = 0
    for category_letter in ALPHABET.lower():
        for no_propagation in [True, False]:
            for validation_run in [True, False]:
                for job in jobs:
                    extra_arguments = [f"--category_starts_with {category_letter}"]
                    if no_propagation: extra_arguments.append("--no_propagation")
                    if validation_run: extra_arguments.append("--validation_run")
                    if validation_run and category_letter == "c":
                        for object_letter in ALPHABET.lower():
                            job.submit(extra_arguments=extra_arguments + [f"--object_starts_with {object_letter}"])
                            job_count += 1
                    else:
                        job.submit(extra_arguments=extra_arguments)
                        job_count += 1

    print(f"Submitted {job_count} jobs.")
