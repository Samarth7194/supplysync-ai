"""Repository package.

Repositories own database access. FastAPI route handlers should call services,
and services should depend on repositories rather than importing SQLAlchemy
models directly.
"""

