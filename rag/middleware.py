from django.http import HttpResponse


class HealthCheckMiddleware:
    """Answer /healthz before host and SSL checks.

    Health checks can arrive over plain HTTP with an unexpected Host header,
    which ALLOWED_HOSTS and SECURE_SSL_REDIRECT would otherwise reject.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path == "/healthz":
            return HttpResponse("ok")
        return self.get_response(request)