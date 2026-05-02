"""Future FastAPI entry point for GroverLab."""

from groverlab.grover_education import educational_overview


def create_app():
    """Create and configure the FastAPI app."""

    from fastapi import FastAPI

    app = FastAPI(title="GroverLab API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/education")
    def education() -> dict:
        return educational_overview()

    return app


app = create_app()

