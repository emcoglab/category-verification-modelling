from copy import deepcopy
from pathlib import Path

from framework.cli.job import VocabEvolutionCategoryVerificationJob, VocabEvolutionCategoryVerificationJobSpec
from framework.evolution.corpora import FILTERED_CORPORA


class Job_VE_1(VocabEvolutionCategoryVerificationJob):

    # max_sphere_radius (i.e. pruning distance) -> RAM/G
    @classmethod
    def SM_RAM(cls, distance: float) -> int:
        if distance <= 1:
            return 6
        elif distance <= 1.5:
            return 31
        # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        elif distance <= 1.98:
            return 56
        elif distance <= 2:
            return 62
        else:
            # Max
            return 120

    @classmethod
    def LING_RAM(cls, model: str, words: int) -> int:
        if model == "pmi_ngram":
            # It's probably not actually double, but it's a decent upper-bound
            return 2 * cls.LING_RAM("ppmi_ngram", words=words)
        elif model == "ppmi_ngram":
            if words <=    1_000: return 3
            elif words <=  3_000: return 4
            elif words <= 10_000: return 6
            elif words <= 30_000: return 8
            elif words <= 40_000: return 10
            elif words <= 60_000: return 12
        raise NotImplementedError()

    def __init__(self, spec: VocabEvolutionCategoryVerificationJobSpec):
        super().__init__(
            script_number="ve1",
            script_name="ve_1_modelling.py",
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
    for s in VocabEvolutionCategoryVerificationJobSpec.load_multiple(
            Path(Path(__file__).parent, "job_specifications/2023-01-12 Paper output.yaml")
            # Only run the first job from this spec sheet - that's the main one, no the no-prop one
            )[:1]:
        for _, corpus in FILTERED_CORPORA.items():
            spec = deepcopy(s)
            spec.replace_corpus(corpus.name)
            jobs.append(Job_VE_1(spec))
        # Also add in the unmodified corpus, just to check
        jobs.append(Job_VE_1(s))

    job_count = 0
    for validation_run in [True, False]:
        for category_letter in ALPHABET.lower():
            for job in jobs:
                extra_arguments = [f"--category_starts_with {category_letter}"]
                if validation_run: extra_arguments.append("--validation_run")
                if validation_run and category_letter == "c":
                    for object_letter in ALPHABET.lower():
                        job.run_locally(extra_arguments=extra_arguments + [f"--object_starts_with {object_letter}"])
                        job_count += 1
                else:
                    job.run_locally(extra_arguments=extra_arguments)
                    job_count += 1

    print(f"Submitted {job_count} jobs.")
