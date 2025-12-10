use std::sync::Arc;

use rocket::{get, post, State};
use rocket::serde::json::Json;

use crate::model_registry::{ModelRegistry, ModelStatus};
use crate::types::{
    HealthResponse,
    InferRequest,
    InferResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfoResponse,
};

#[get("/health")]
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

#[get("/models")]
pub async fn list_models(
    registry: &State<Arc<ModelRegistry>>,
) -> Json<Vec<ModelInfoResponse>> {
    let models = registry.list_models();
    let resp: Vec<ModelInfoResponse> = models
        .into_iter()
        .map(|m| ModelInfoResponse {
            name: m.name,
            status: format!("{:?}", m.status),
        })
        .collect();

    Json(resp)
}

#[post("/load", data = "<req>")]
pub async fn load_model(
    registry: &State<Arc<ModelRegistry>>,
    req: Json<LoadModelRequest>,
) -> Json<LoadModelResponse> {
    let model_name = &req.model_name;

    // Sprint 1: 只是修改状态，不做真实加载
    let maybe_meta = registry.set_status(model_name, ModelStatus::Loaded);

    match maybe_meta {
        Some(meta) => Json(LoadModelResponse {
            model_name: meta.name,
            status: format!("{:?}", meta.status),
            message: "model marked as Loaded (dummy)".to_string(),
        }),
        None => Json(LoadModelResponse {
            model_name: model_name.clone(),
            status: "NotFound".to_string(),
            message: "model not found in registry".to_string(),
        }),
    }
}

#[post("/infer", data = "<req>")]
pub async fn infer(
    registry: &State<Arc<ModelRegistry>>,
    req: Json<InferRequest>,
) -> Json<InferResponse> {
    let model_name = &req.model_name;

    // 检查模型是否存在且已 Loaded
    let meta = registry.get_model(model_name);
    if meta.is_none() {
        return Json(InferResponse {
            model_name: model_name.clone(),
            output: format!("Error: model `{}` not found", model_name),
        });
    }
    let meta = meta.unwrap();
    if !matches!(meta.status, ModelStatus::Loaded) {
        return Json(InferResponse {
            model_name: model_name.clone(),
            output: format!(
                "Error: model `{}` is not loaded (status = {:?})",
                model_name, meta.status
            ),
        });
    }

    // Sprint 1: dummy 推理，直接 echo
    let output = format!("[{}] echo: {}", model_name, req.prompt);

    Json(InferResponse {
        model_name: model_name.clone(),
        output,
    })
}
