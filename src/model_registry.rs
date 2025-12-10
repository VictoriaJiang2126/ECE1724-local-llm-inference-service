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

#[derive(Debug, Clone, Serialize)]
pub struct ModelMetadata {
    pub name: String,
    pub status: ModelStatus,
    pub path: String,          // 将来：模型权重路径
    pub quantization: String,  // 将来：q4/q8 等
    pub last_updated: Option<SystemTime>,
}

impl ModelMetadata {
    pub fn new(name: &str, path: &str, quantization: &str) -> Self {
        Self {
            name: name.to_string(),
            status: ModelStatus::Unloaded,
            path: path.to_string(),
            quantization: quantization.to_string(),
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
        // 这里先硬编码一些模型，后续可以从配置文件/环境变量加载
        let mut map = HashMap::new();
        map.insert(
            "mistral-7b".to_string(),
            ModelMetadata::new("mistral-7b", "./models/mistral-7b", "q4_k_m"),
        );
        map.insert(
            "llama-3b".to_string(),
            ModelMetadata::new("llama-3b", "./models/llama-3b", "q4_k_m"),
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
