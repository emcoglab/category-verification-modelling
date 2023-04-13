from copy import deepcopy
from pathlib import Path

from framework.cli.job import VocabEvolutionCategoryVerificationJob, VocabEvolutionCategoryVerificationJobSpec
from framework.evolution.corpora import FILTERED_CORPORA


class Job_VE_1(VocabEvolutionCategoryVerificationJob):

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

    def LING_RAM(self, model: str, words: int) -> int:
        if model == "pmi_ngram":
            if words <=    1_000: return 2
            elif words <=  3_000: return 3
            elif words <= 10_000: return 7
            elif words <= 30_000: return 11
            elif words <= 40_000: return 15
            elif words <= 60_000: return 20
        elif model == "ppmi_ngram":
            if words <=    1_000: return 2
            elif words <=  3_000: return 3
            elif words <= 10_000: return 5
            elif words <= 30_000: return 7
            elif words <= 40_000: return 9
            elif words <= 60_000: return 11
        raise NotImplementedError()

    def __init__(self, spec: VocabEvolutionCategoryVerificationJobSpec):
        super().__init__(
            script_number="1",
            script_name="1_modelling.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, VocabEvolutionCategoryVerificationJobSpec)
        return (self.SM_RAM(self.spec.sensorimotor_spec.max_radius)
                + self.LING_RAM(model=self.spec.linguistic_spec.model_name, words=self.spec.linguistic_spec.n_words))


if __name__ == '__main__':

    ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    jobs = []
    s: VocabEvolutionCategoryVerificationJobSpec
    for s in VocabEvolutionCategoryVerificationJobSpec.load_multiple(Path(Path(__file__).parent,
                                                            "job_specifications/2023-01-12 Paper output.yaml")):
        for _, corpus in FILTERED_CORPORA.items():
            spec = deepcopy(s)
            spec.linguistic_spec.corpus_name = corpus.name
            jobs.append(Job_VE_1(spec))

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
                            job.run_locally(extra_arguments=extra_arguments + [f"--object_starts_with {object_letter}"])
                            job_count += 1
                    else:
                        job.run_locally(extra_arguments=extra_arguments)
                        job_count += 1

    print(f"Submitted {job_count} jobs.")
