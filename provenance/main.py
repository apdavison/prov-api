from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware

from . import (
    settings,
    simulation,
    dataanalysis,
    visualisation,
    optimisation,
    datacopy,
    generic,
    workflows,
    recipes,
    datasets,
    auth,
    statistics,
)

description = """
This is a first release candidate, more testing is needed before the first release.

To use the API, <a href="/login" target="_blank">login here</a>, click on "Authorize" then
copy the <i>access_token</i> into the "HTTPBearer" box
(this process will be streamlined for the beta release).
"""

app = FastAPI(title="EBRAINS Provenance API", description=description, version="1.0")

app.add_middleware(SessionMiddleware, secret_key=settings.SESSIONS_SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recipes.router, tags=["Workflow Recipes"])
app.include_router(workflows.router, tags=["Workflow Executions"])
app.include_router(statistics.router, tags=["Statistics"])
app.include_router(simulation.router, tags=["Simulations"])
app.include_router(dataanalysis.router, tags=["Data analysis"])
app.include_router(visualisation.router, tags=["Visualisation"])
app.include_router(optimisation.router, tags=["Optimisation"])
app.include_router(datacopy.router, tags=["Data transfer"])
app.include_router(generic.router, tags=["Unclassified"])
app.include_router(datasets.router, tags=["Dataset generation"])
app.include_router(auth.router, tags=["Authentication and authorization"])
#app.include_router(vocab.router, tags=["Controlled vocabularies"])
