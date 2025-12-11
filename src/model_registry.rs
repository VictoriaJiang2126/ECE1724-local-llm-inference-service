use std::collections::HashMap;
use std::time::SystemTime;

use parking_lot::RwLock;
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub enum ModelStatus {
    Unloaded,
    Loading,
    Loaded,
    Error,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum EngineKind {
    Dummy,
    Candle, // 以后可以打开这一行
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelMetadata {
    pub name: String,
    pub status: ModelStatus,
    pub path: String,
    pub quantization: String,
    pub engine_kind: EngineKind,
    pub last_updated: Option<SystemTime>,
}

impl ModelMetadata {
    pub fn new(
        name: &str,
        path: &str,
        quantization: &str,
        engine_kind: EngineKind,
    ) -> Self {
        Self {
            name: name.to_string(),
            status: ModelStatus::Unloaded,
            path: path.to_string(),
            quantization: quantization.to_string(),
            engine_kind,
            last_updated: None,
        }
    }
}

#[derive(Debug)]
pub struct ModelRegistry {
    pub models: RwLock<HashMap<String, ModelMetadata>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        map.insert(
            "mistral-7b".to_string(),
            ModelMetadata::new(
                "mistral-7b",
                "./models/mistral-7b",
                "q4_k_m",
                EngineKind::Candle,
            ),
        );
        map.insert(
            "llama-3b".to_string(),
            ModelMetadata::new(
                "llama-3b",
                "./models/llama-3b",
                "q4_k_m",
                EngineKind::Dummy,
            ),
        );

        Self {
            models: RwLock::new(map),
        }
    }

    pub fn list_models(&self) -> Vec<ModelMetadata> {
        let guard = self.models.read();
        guard.values().cloned().collect()
    }

    pub fn set_status(&self, name: &str, status: ModelStatus) -> Option<ModelMetadata> {
        let mut guard = self.models.write();
        if let Some(meta) = guard.get_mut(name) {
            meta.status = status;
            meta.last_updated = Some(SystemTime::now());
            return Some(meta.clone());
        }
        None
    }

    pub fn get_model(&self, name: &str) -> Option<ModelMetadata> {
        let guard = self.models.read();
        guard.get(name).cloned()
    }
}
