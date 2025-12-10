use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use rocket::tokio::sync::mpsc;

/// 统一的推理引擎抽象
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// 一次性生成完整结果
    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;

    /// 流式生成：把结果按 chunk 推送到 sender 中
    async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        sender: mpsc::Sender<String>,
    ) -> Result<()>;
}

/// Dummy 实现：只做字符串处理和延迟模拟
pub struct DummyEngine {
    pub model_name: String,
}

impl DummyEngine {
    pub fn new(model_name: &str) -> Arc<Self> {
        Arc::new(Self {
            model_name: model_name.to_string(),
        })
    }
}

#[async_trait]
impl InferenceEngine for DummyEngine {
    async fn generate(&self, prompt: &str, _max_tokens: usize) -> Result<String> {
        // 模拟一点延迟
        rocket::tokio::time::sleep(Duration::from_millis(50)).await;

        let output = format!(
            "[{} DUMMY] {}",
            self.model_name,
            prompt.to_uppercase()
        );
        Ok(output)
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        _max_tokens: usize,
        sender: mpsc::Sender<String>,
    ) -> Result<()> {
        // 一样生成最终输出，但按“词”切片发送
        let full = format!(
            "[{} DUMMY] {}",
            self.model_name,
            prompt.to_uppercase()
        );

        let mut words: Vec<String> = full
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        // 最前面加一个“模型名”chunk 方便前端展示
        words.insert(0, format!("[model={}]", self.model_name));

        for w in words {
            if sender.send(w.clone()).await.is_err() {
                // 客户端断开连接
                break;
            }
            rocket::tokio::time::sleep(Duration::from_millis(50)).await;
        }

        Ok(())
    }
}
