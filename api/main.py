"""FastAPI application exposing the Crisis Command multi-agent workflow."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from agents.safeguards import PromptInjectionDetector
from agents.schemas import IncidentSubmission, Severity
from api.schemas import HealthStatus, IncidentDetail, IncidentList, IncidentSummary, InjectionRejection
from config import PROJECT_ROOT, get_settings
from db.database import get_database, get_db
from db.models import Incident, IncidentStatus
from graph.workflow import CrisisWorkflow, get_workflow

logger = logging.getLogger("crisis_command.api")


def configure_logging(level: str) -> None:
    """Configure root logging once for the application.

    Args:
        level: Log level name, e.g. ``INFO``.
    """
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise the database schema, agents and knowledge base on startup.

    Args:
        app: The FastAPI application.

    Yields:
        Control back to FastAPI while the app is running.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    get_database().create_all()
    workflow = get_workflow()
    if settings.auto_ingest and workflow.knowledge_base is not None:
        from knowledge_base.ingest import ingest_if_empty

        counts = ingest_if_empty(workflow.knowledge_base)
        if counts:
            logger.info("Knowledge base was empty; ingested seed data %s", counts)
    app.state.detector = PromptInjectionDetector(settings.injection_threshold, settings.max_report_chars)
    logger.info("Crisis Command API ready (model=%s)", settings.groq_model)
    yield


app = FastAPI(
    title="Critical Command Crisis Center",
    description="Multi-agent AI crisis triage: intake -> analysis -> RAG -> response strategy.",
    version="1.0.0",
    lifespan=lifespan,
)

# Lets the browser dashboard call the API when opened from file:// or another origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allow_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

DASHBOARD_FILE = PROJECT_ROOT / "dashboard.html"


@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the single-file command dashboard.

    Returns:
        The dashboard HTML.

    Raises:
        HTTPException: 404 if ``dashboard.html`` is missing.
    """
    if not DASHBOARD_FILE.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="dashboard.html not found")
    return FileResponse(DASHBOARD_FILE, media_type="text/html")


def workflow_dependency() -> CrisisWorkflow:
    """FastAPI dependency returning the workflow singleton."""
    return get_workflow()


def detector_dependency(request: Request) -> PromptInjectionDetector:
    """FastAPI dependency returning the shared prompt-injection detector."""
    return request.app.state.detector


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic 500 without leaking internals, while logging the traceback.

    Args:
        request: The failing request.
        exc: The unhandled exception.

    Returns:
        A JSON 500 response.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthStatus, tags=["system"])
def health(db: Session = Depends(get_db), workflow: CrisisWorkflow = Depends(workflow_dependency)) -> HealthStatus:
    """Report database and knowledge-base health.

    Args:
        db: Database session.
        workflow: Workflow service.

    Returns:
        Health information.
    """
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {type(exc).__name__}"
    kb = workflow.knowledge_base.counts() if workflow.knowledge_base is not None else "unavailable"
    overall = "ok" if db_status == "ok" else "degraded"
    return HealthStatus(status=overall, database=db_status, knowledge_base=kb, model=get_settings().groq_model)


@app.post(
    "/incident",
    response_model=IncidentDetail,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": InjectionRejection}, 202: {"model": IncidentDetail}},
    tags=["incidents"],
)
def submit_incident(
    payload: IncidentSubmission,
    response: Response,
    background_tasks: BackgroundTasks,
    run_async: bool = Query(False, description="Return immediately (202) and process in the background."),
    db: Session = Depends(get_db),
    workflow: CrisisWorkflow = Depends(workflow_dependency),
    detector: PromptInjectionDetector = Depends(detector_dependency),
) -> IncidentDetail | JSONResponse:
    """Submit a new incident and run the full multi-agent LangGraph workflow.

    Args:
        payload: Validated incident submission.
        response: Response object (used to set 202 in async mode).
        background_tasks: FastAPI background task queue.
        run_async: Whether to process asynchronously.
        db: Database session.
        workflow: Workflow service.
        detector: Prompt-injection detector.

    Returns:
        The stored incident with all agent outputs (or the RECEIVED record in async mode).
    """
    screened = [detector.check(v) for v in (payload.report, payload.title, payload.location) if v]
    flagged = [g for g in screened if g.is_injection]
    if flagged:
        worst = max(flagged, key=lambda g: g.risk_score)
        logger.warning("Rejected submission: prompt injection (score=%.2f, rules=%s)", worst.risk_score, worst.matched_rules)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=InjectionRejection(
                detail="Report rejected: possible prompt-injection content detected.",
                risk_score=worst.risk_score,
                matched_rules=worst.matched_rules,
            ).model_dump(),
        )

    incident = workflow.create_incident(db, payload)

    if run_async:
        background_tasks.add_task(workflow.process_incident, incident.id)
        response.status_code = status.HTTP_202_ACCEPTED
        return IncidentDetail.model_validate(incident)

    workflow.process_incident(incident.id)
    db.expire_all()
    stored = db.get(Incident, incident.id)
    return IncidentDetail.model_validate(stored)


@app.get("/incidents", response_model=IncidentList, tags=["incidents"])
def list_incidents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    severity: Severity | None = Query(None, description="Filter by severity."),
    status_filter: IncidentStatus | None = Query(None, alias="status", description="Filter by status."),
    db: Session = Depends(get_db),
) -> IncidentList:
    """List incidents, newest first.

    Args:
        skip: Offset for pagination.
        limit: Page size.
        severity: Optional severity filter.
        status_filter: Optional status filter.
        db: Database session.

    Returns:
        A page of incident summaries.
    """
    query = select(Incident)
    if severity is not None:
        query = query.where(Incident.severity == severity.value)
    if status_filter is not None:
        query = query.where(Incident.status == status_filter.value)
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.scalars(query.order_by(Incident.created_at.desc()).offset(skip).limit(limit)).all()
    return IncidentList(
        total=total, skip=skip, limit=limit, items=[IncidentSummary.model_validate(r) for r in rows]
    )


@app.get("/incidents/{incident_id}", response_model=IncidentDetail, tags=["incidents"])
def get_incident(incident_id: str, db: Session = Depends(get_db)) -> IncidentDetail:
    """Get one incident with the full multi-agent response.

    Args:
        incident_id: Incident UUID.
        db: Database session.

    Returns:
        The incident detail.

    Raises:
        HTTPException: 404 if the incident does not exist.
    """
    incident = db.get(Incident, incident_id)
    if incident is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")
    return IncidentDetail.model_validate(incident)
