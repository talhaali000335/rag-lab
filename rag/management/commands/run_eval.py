import os

from django.core.management.base import BaseCommand, CommandError

from rag.eval_set import GOLDEN
from rag.models import EvalRun
from rag.pipelines import PIPELINES
from rag.services import ask


class Command(BaseCommand):
    help = "Run the golden question set against every pipeline; fail if hit-rate drops (use as a CI gate)."

    def add_arguments(self, parser):
        parser.add_argument("--min-hit-rate", type=float, default=0.8)
        parser.add_argument("--pipelines", nargs="*", default=["advanced", "corrective", "agentic", "multimodal", "adaptive"])

    def handle(self, *args, **opts):
        failed = []
        for slug in opts["pipelines"]:
            hits, ground = 0, 0.0
            for question, expected in GOLDEN:
                state = ask(question, slug, save=False)
                hits += expected in {d.id for d in state.get("sources", [])}
                ground += state.get("groundedness", 0)
            n = len(GOLDEN)
            run = EvalRun.objects.create(pipeline=slug, hit_rate=hits / n, avg_groundedness=ground / n, cases=n,
                                         git_sha=os.environ.get("GIT_SHA", ""))
            self.stdout.write(f"{PIPELINES[slug]['name']:<24} hit-rate={run.hit_rate:.2f} groundedness={run.avg_groundedness:.2f}")
            if run.hit_rate < opts["min_hit_rate"]:
                failed.append(slug)
        if failed:
            raise CommandError(f"Quality gate failed for: {', '.join(failed)}")
