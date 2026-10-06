from django.db import models


class QueryLog(models.Model):
    """One row per question — the raw material for monitoring and evaluation (MLOps)."""
    created = models.DateTimeField(auto_now_add=True, db_index=True)
    pipeline = models.CharField(max_length=20, db_index=True)
    role = models.CharField(max_length=10, default="public")
    question = models.CharField(max_length=500)  # already PII-redacted by the input guardrail
    answer = models.TextField(blank=True)
    blocked = models.BooleanField(default=False)
    findings = models.CharField(max_length=200, blank=True)
    groundedness = models.FloatField(default=0)
    latency_ms = models.IntegerField(default=0)
    source_ids = models.CharField(max_length=100, blank=True)

    class Meta:
        ordering = ["-created"]


class EvalRun(models.Model):
    """Result of `manage.py run_eval` for one pipeline — track quality across deployments."""
    created = models.DateTimeField(auto_now_add=True, db_index=True)
    pipeline = models.CharField(max_length=20)
    hit_rate = models.FloatField()
    avg_groundedness = models.FloatField()
    cases = models.IntegerField()
    git_sha = models.CharField(max_length=40, blank=True)

    class Meta:
        ordering = ["-created"]
