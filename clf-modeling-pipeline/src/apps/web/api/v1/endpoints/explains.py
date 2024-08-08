from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import Response
from src.services.models.inspection import get_shap_global_plots

router = APIRouter(prefix="/explains", tags=["explains"])

@router.get("/info")
def explain_info() -> str:
    return "Hello, Shap Explainer here"

@router.get("/global_plots")
def _get_shap_global_plots(model_id: str) -> Response:
    bar_plot = get_shap_global_plots(
        'test-registry-1234', 
        model_id='test-mlflow-sgd', 
        use_train=True
    )
    return Response(content=bar_plot.getvalue(), media_type="image/png")