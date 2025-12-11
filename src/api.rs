use std::sync::Arc;

use rocket::{get, post, Shutdown, State};
use rocket::response::stream::{Event, EventStream};
use rocket::serde::json::Json;
use rocket::tokio::select;
use rocket::tokio::sync::mpsc;

use crate::app_state::AppState;
use crate::model_registry::ModelStatus;
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
    state: &State<Arc<AppState>>,
) -> Json<Vec<ModelInfoResponse>> {
    let models = state.list_models();
    let resp: Vec<ModelInfoResponse> = models
        .into_iter()
        .map(|m| ModelInfoResponse {
            name: m.name,
            status: format!("{:?}", m.status),
            engine_kind: format!("{:?}", m.engine_kind),
        })
        .collect();

    Json(resp)
}

#[post("/load", data = "<req>")]
pub async fn load_model(
    state: &State<Arc<AppState>>,
    req: Json<LoadModelRequest>,
) -> Json<LoadModelResponse> {
    let model_name = &req.model_name;

    match state.load_model(model_name) {
        Ok(meta) => Json(LoadModelResponse {
            model_name: meta.name,
            status: format!("{:?}", meta.status),
            message: "model loaded (DummyEngine)".to_string(),
        }),
        Err(e) => Json(LoadModelResponse {
            model_name: model_name.clone(),
            status: "Error".to_string(),
            message: e,
        }),
    }
}

/// 非流式：POST /infer
#[post("/infer", data = "<req>", rank = 2)]
pub async fn infer(
    state: &State<Arc<AppState>>,
    req: Json<InferRequest>,
) -> Json<InferResponse> {
    let model_name = &req.model_name;

    let meta = state.registry.get_model(model_name);
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

    let engine = state.get_engine(model_name);
    if engine.is_none() {
        return Json(InferResponse {
            model_name: model_name.clone(),
            output: format!("Error: no engine instance for model `{}`", model_name),
        });
    }
    let engine = engine.unwrap();

    let permit = state.semaphore.clone().acquire_owned().await.unwrap();

    let prompt = req.prompt.clone();
    let result = engine.generate(&prompt, 64).await;

    drop(permit);

    let output = match result {
        Ok(text) => text,
        Err(e) => format!("Error during inference: {}", e),
    };

    Json(InferResponse {
        model_name: model_name.clone(),
        output,
    })
}

/// 流式 SSE：POST /infer?stream=true
#[post("/infer?<stream>", data = "<req>", rank = 1)]
pub async fn infer_stream(
    state: &State<Arc<AppState>>,
    req: Json<InferRequest>,
    stream: bool,
    mut shutdown: Shutdown,
) -> EventStream![] {
    let state = state.inner().clone(); // Arc<AppState>
    let model_name = req.model_name.clone();
    let prompt = req.prompt.clone();

    EventStream! {
        if !stream {
            // 情况 1：没带 stream=true
            yield Event::data("stream=false not supported on this endpoint");
            return;
        }

        // 情况 2：检查模型是否存在
        let meta_opt = state.registry.get_model(&model_name);
        if meta_opt.is_none() {
            yield Event::data(format!("Error: model `{}` not found", model_name));
            return;
        }
        let meta = meta_opt.unwrap();
        if !matches!(meta.status, ModelStatus::Loaded) {
            yield Event::data(format!(
                "Error: model `{}` is not loaded (status = {:?})",
                model_name, meta.status
            ));
            return;
        }

        // 情况 3：检查 engine 是否存在
        let engine_opt = state.get_engine(&model_name);
        if engine_opt.is_none() {
            yield Event::data(format!("Error: no engine instance for `{}`", model_name));
            return;
        }
        let engine = engine_opt.unwrap();

        // 获取 semaphore permit，控制并发
        let semaphore = state.semaphore.clone();
        let permit = semaphore.acquire_owned().await.unwrap();

        // 建立 channel
        let (tx, mut rx) = mpsc::channel::<String>(32);

        // 后台任务：调用 engine.generate_stream
        rocket::tokio::spawn(async move {
            let _permit = permit; // 生命周期结束自动释放
            let _ = engine.generate_stream(&prompt, 128, tx).await;
        });

        // 真正的 SSE 主循环
        loop {
            select! {
                maybe_chunk = rx.recv() => {
                    match maybe_chunk {
                        Some(text) => {
                            // 每个 chunk 一个 SSE 事件
                            yield Event::data(text);
                        }
                        None => {
                            // 生成结束
                            break;
                        }
                    }
                }
                _ = &mut shutdown => {
                    // 客户端断开 或 服务器关闭
                    break;
                }
            }
        }
    }
}


/// GET SSE：/infer_stream?model_name=xxx&prompt=yyy
#[get("/infer_stream?<model_name>&<prompt>")]
pub async fn infer_stream_get(
    state: &State<Arc<AppState>>,
    model_name: &str,
    prompt: &str,
    mut shutdown: Shutdown,
) -> EventStream![] {
    let state = state.inner().clone();
    let model_name = model_name.to_string();
    let prompt = prompt.to_string();

    EventStream! {
        // 1) 校验模型是否存在 & 已加载
        let meta_opt = state.registry.get_model(&model_name);
        if meta_opt.is_none() {
            yield Event::data(format!("Error: model `{}` not found", model_name));
            return;
        }
        let meta = meta_opt.unwrap();
        if !matches!(meta.status, ModelStatus::Loaded) {
            yield Event::data(format!(
                "Error: model `{}` is not loaded (status = {:?})",
                model_name, meta.status
            ));
            return;
        }

        // 2) 获取 engine
        let engine_opt = state.get_engine(&model_name);
        if engine_opt.is_none() {
            yield Event::data(format!("Error: no engine instance for `{}`", model_name));
            return;
        }
        let engine = engine_opt.unwrap();

        // 3) 并发控制
        let semaphore = state.semaphore.clone();
        let permit = semaphore.acquire_owned().await.unwrap();

        // 4) 建 channel
        let (tx, mut rx) = mpsc::channel::<String>(32);

        // 5) 后台推理任务（流式写入 tx）
        rocket::tokio::spawn(async move {
            let _permit = permit; // 保证推理期间占用 slot
            let _ = engine.generate_stream(&prompt, 128, tx).await;
        });

        // 6) 主循环：把 channel 里的 chunk 以 SSE 事件发给前端
        loop {
            select! {
                maybe_chunk = rx.recv() => {
                    match maybe_chunk {
                        Some(text) => {
                            yield Event::data(text);
                        }
                        None => {
                            break;
                        }
                    }
                }
                _ = &mut shutdown => {
                    break;
                }
            }
        }
    }
}