use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use rocket::tokio::sync::mpsc;

// Candle 相关
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama as qllama;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer; // ✅ 用 candle_core

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

        let output = format!("[{} DUMMY] {}", self.model_name, prompt.to_uppercase());
        Ok(output)
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        _max_tokens: usize,
        sender: mpsc::Sender<String>,
    ) -> Result<()> {
        // 一样生成最终输出，但按“词”切片发送
        let full = format!("[{} DUMMY] {}", self.model_name, prompt.to_uppercase());

        let mut words: Vec<String> = full.split_whitespace().map(|s| s.to_string()).collect();

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

use std::sync::Mutex;
pub struct CandleEngine {
    model_name: String,
    device: Device,
    model: Mutex<qllama::ModelWeights>,
    tokenizer: Tokenizer,
}

impl CandleEngine {
    pub fn new(model_name: &str) -> anyhow::Result<Arc<Self>> {
        // 1) 设备：先用 CPU，后面你可以改成 metal/cuda
        let device = Device::Cpu;

        // 2) 通过 hf-hub 下载 GGUF 权重
        let repo = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF";
        let filename = "mistral-7b-instruct-v0.1.Q2_K.gguf";

        let api = Api::new()?;
        let api = api.model(repo.to_string());
        let model_path = api.get(filename)?;

        let mut file = std::fs::File::open(&model_path)?;
        let start = std::time::Instant::now();

        let content = gguf_file::Content::read(&mut file)?;
        let mut total_size_in_bytes = 0usize;
        for (_, tensor) in content.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        println!(
            "[Candle] loaded {} tensors ({}) in {:.2}s",
            content.tensor_infos.len(),
            format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );

        let model = qllama::ModelWeights::from_gguf(content, &mut file, &device)?;
        println!("[Candle] model built for {}", model_name);

        // 3) 下载 tokenizer
        let api = Api::new()?;
        let repo_tok = "mistralai/Mistral-7B-v0.1";
        let api = api.model(repo_tok.to_string());
        let tokenizer_path = api.get("tokenizer.json")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Error loading tokenizer: {e}"))?;

        Ok(Arc::new(Self {
            model_name: model_name.to_string(),
            device,
            model: Mutex::new(model),
            tokenizer,
        }))
    }

    /// 简单的 greedy / 有温度采样，这里做一个“非流式”生成
    fn generate_inner(&self, prompt: &str, max_tokens: usize) -> anyhow::Result<String> {
        let sample_len: usize = max_tokens;
        let temperature: f64 = 0.8;
        let top_p: Option<f64> = None;
        let seed: u64 = 42;
        // 目前没用到，可先注释掉或前缀 _
        // let repeat_penalty: f32 = 1.1;
        // let repeat_last_n: usize = 64;

        let temperature = if temperature == 0.0 {
            None
        } else {
            Some(temperature)
        };

        let prompt_str = format!("[INST] {prompt} [/INST]");
        let tokens = self
            .tokenizer
            .encode(prompt_str, true)
            .map_err(|e| anyhow::anyhow!("Error encoding tokenizer: {e}"))?;
        let mut prompt_tokens = tokens.get_ids().to_vec();
        let to_sample = sample_len.saturating_sub(1);

        if prompt_tokens.len() + to_sample > qllama::MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - qllama::MAX_SEQ_LEN;
            prompt_tokens = prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec();
        }

        let mut all_tokens = vec![];
        let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);

        //  关键：从 Mutex 中拿一个可变的 model 引用
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("failed to lock model mutex"))?;

        // 1) 先跑 prompt
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = model.forward(&input, 0)?; // ✅ 用可变 model
        logits = logits.squeeze(0)?;
        let mut next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);

        let eos_token = *self.tokenizer.get_vocab(true).get("</s>").unwrap_or(&0);

        // 2) 继续采样
        for _ in 0..to_sample {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?.squeeze(0)?;
            next_token = logits_processor.sample(&logits)?;
            if next_token == eos_token {
                break;
            }
            all_tokens.push(next_token);
        }

        // 3) decode 回字符串
        let mut out_tokens = prompt_tokens.clone();
        out_tokens.extend(all_tokens.iter());
        let decoded = self
            .tokenizer
            .decode(&out_tokens, true)
            .map_err(|e| anyhow::anyhow!("Error decoding: {e}"))?;

        Ok(decoded)
    }
}

// 小工具：人类可读的字节数
fn format_size(size: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let size_f = size as f64;
    if size_f > GB {
        format!("{:.2} GiB", size_f / GB)
    } else if size_f > MB {
        format!("{:.2} MiB", size_f / MB)
    } else if size_f > KB {
        format!("{:.2} KiB", size_f / KB)
    } else {
        format!("{size} B")
    }
}

#[async_trait]
impl InferenceEngine for CandleEngine {
    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let out = self.generate_inner(prompt, max_tokens)?;
        Ok(out)
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        sender: mpsc::Sender<String>,
    ) -> Result<()> {
        let full = self.generate(prompt, max_tokens).await?;
        for w in full.split_whitespace() {
            if sender.send(w.to_string()).await.is_err() {
                break;
            }
            rocket::tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        }
        Ok(())
    }
}
