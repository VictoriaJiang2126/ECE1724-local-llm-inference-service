use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::sync::Semaphore;

use crate::engine::{DummyEngine, InferenceEngine};
use crate::model_registry::{EngineKind, ModelMetadata, ModelRegistry, ModelStatus};



/// 全局共享状态：
/// - registry: 记录模型元信息和状态
/// - engines: model_name -> 对应 InferenceEngine 实例
/// - semaphore: 控制最多 N 个并发推理任务

pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    pub engines: RwLock<HashMap<String, Arc<dyn InferenceEngine>>>,
    pub semaphore: Arc<Semaphore>,
    pub max_concurrent_infer: usize,
}
impl AppState {
    pub fn new(max_concurrent_infer: usize) -> Arc<Self> {
        Arc::new(Self {
            registry: Arc::new(ModelRegistry::new()),
            engines: RwLock::new(HashMap::new()),
            semaphore: Arc::new(Semaphore::new(max_concurrent_infer)),
            max_concurrent_infer,
        })
    }

    pub fn list_models(&self) -> Vec<ModelMetadata> {
        self.registry.list_models()
    }

    /// 加载模型：根据 EngineKind 创建对应 Engine，并放入 engines 映射中
    pub fn load_model(&self, model_name: &str) -> Result<ModelMetadata, String> {
        let meta = self
            .registry
            .get_model(model_name)
            .ok_or_else(|| format!("model `{}` not found", model_name))?;

        // 先标记为 Loading
        let _ = self.registry.set_status(model_name, ModelStatus::Loading);

        // 根据 engine_kind 创建具体的 Engine 实例
        let engine: Arc<dyn InferenceEngine> = match meta.engine_kind {
            EngineKind::Dummy => DummyEngine::new(model_name),
            // EngineKind::Candle => { ... 构造 CandleEngine ... }
        };

        {
            let mut guard = self.engines.write();
            guard.insert(model_name.to_string(), engine);
        }

        // 标记为 Loaded
        let meta = self
            .registry
            .set_status(model_name, ModelStatus::Loaded)
            .ok_or_else(|| format!("failed to update status for `{}`", model_name))?;

        Ok(meta)
    }

    /// 获取已加载的 InferenceEngine
    pub fn get_engine(&self, model_name: &str) -> Option<Arc<dyn InferenceEngine>> {
        let guard = self.engines.read();
        guard.get(model_name).cloned()
    }
}
