use anyhow::{Result, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::{LlamaSampler, params::LlamaSamplerChainParams};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex, Once};
use lazy_static::lazy_static;
use std::fs::File;
use std::io::Write;

static INIT: Once = Once::new();

lazy_static! {
    static ref BACKEND: Arc<Mutex<Option<Arc<LlamaBackend>>>> = Arc::new(Mutex::new(None));
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub max_length: usize,
    #[serde(default)]
    pub additional_gen_config: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            repetition_penalty: 1.1,
            max_length: 4096,
            additional_gen_config: std::collections::HashMap::new(),
        }
    }
}

pub struct GGUFModel {
    model: Arc<LlamaModel>,
    context: Arc<Mutex<llama_cpp_2::context::LlamaContext<'static>>>,
    backend: Arc<LlamaBackend>,
}

pub struct ModelOutput {
    audio: Vec<f32>,
    sr: u32,
}

impl ModelOutput {
    pub fn new(audio: Vec<f32>, sr: u32) -> Self {
        Self { audio, sr }
    }
    
    fn to_wav_bytes(&self) -> Vec<u8> {
        let samples = &self.audio;
        let sr = self.sr;
        
        let data_size = (samples.len() * 4) as u32;
        let total_size = 36 + data_size;
        let mut buffer = Vec::with_capacity(44 + samples.len() * 4);
        
        buffer.extend(b"RIFF");
        buffer.extend(&total_size.to_le_bytes());
        buffer.extend(b"WAVE");
        buffer.extend(b"fmt ");
        buffer.extend(&16u32.to_le_bytes());
        buffer.extend(&3u16.to_le_bytes());
        buffer.extend(&1u16.to_le_bytes());
        buffer.extend(&sr.to_le_bytes());
        buffer.extend(&(sr * 4).to_le_bytes());
        buffer.extend(&4u16.to_le_bytes());
        buffer.extend(&32u16.to_le_bytes());
        buffer.extend(b"data");
        buffer.extend(&data_size.to_le_bytes());
        
        for sample in samples {
            buffer.extend(&sample.to_le_bytes());
        }
        
        buffer
    }
    
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let wav_data = self.to_wav_bytes();
        let mut file = File::create(path)?;
        file.write_all(&wav_data)?;
        Ok(())
    }
}

impl GGUFModel {
    pub fn new(
        model_path: impl AsRef<Path>,
        n_gpu_layers: u32,
        max_seq_length: usize,
        additional_model_config: Option<std::collections::HashMap<String, serde_json::Value>>,
    ) -> Result<Self> {
        let backend = {
            let mut backend_guard = BACKEND.lock().unwrap();
            if backend_guard.is_some() {
                backend_guard.as_ref().unwrap().clone()
            } else {
                let new_backend = Arc::new(LlamaBackend::init()?);
                *backend_guard = Some(new_backend.clone());
                INIT.call_once(|| {});
                new_backend
            }
        };

        let mut model_params = LlamaModelParams::default();
        
        // Apply additional configuration if provided
        if let Some(config) = additional_model_config {
            if let Some(n_gpu_layers) = config.get("n_gpu_layers").and_then(|v| v.as_i64()) {
                model_params = model_params.with_n_gpu_layers(n_gpu_layers as u32);
            }
        }

        let model = Arc::new(LlamaModel::load_from_file(&backend, model_path, &model_params)?);
        
        let ctx_size = NonZeroU32::new(max_seq_length as u32)
            .ok_or_else(|| anyhow::anyhow!("Context size must be greater than zero"))?;
        
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size))
            .with_n_threads_batch(8)
            .with_n_threads(8);
            
        let context = unsafe {
            Arc::new(Mutex::new(std::mem::transmute::<
                llama_cpp_2::context::LlamaContext<'_>,
                llama_cpp_2::context::LlamaContext<'static>
            >(model.new_context(&backend, ctx_params)?)))
        };

        Ok(Self { model, context, backend })
    }

    pub fn generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        stream: bool,
    ) -> Result<ModelOutput> {
        if stream {
            self.generate_stream(prompt, config)
        } else {
            self.generate_sync(prompt, config)
        }
    }

    fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<ModelOutput> {
        let mut context = self.context.lock().unwrap();
        let tokens_list = self.model.str_to_token(prompt, AddBos::Always)?;
        
        if tokens_list.len() >= config.max_length {
            bail!("the prompt is too long, it has more tokens than max_length");
        }

        let mut batch = LlamaBatch::new(1024, 1);
        let last_index = tokens_list.len() as i32 - 1;
        
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }
        
        context.decode(&mut batch)?;
        
        let mut n_cur = batch.n_tokens();
        let mut generated_text = String::new();
        
        let mut sampler = LlamaSampler::new(LlamaSamplerChainParams::default())?
            .add_mirostat_v2(42, 5.0, 0.1)
            .add_temp(config.temperature);
            
        while n_cur <= config.max_length as i32 {
            let token = sampler.sample(&context, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
            let output_string = String::from_utf8_lossy(&output_bytes).to_string();
            generated_text.push_str(&output_string);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            context.decode(&mut batch)?;
            
            if n_cur % 1024 == 0 {
                context.clear_kv_cache();
            }
        }

        // Convert bytes to f32 samples
        let samples: Vec<f32> = generated_text.into_bytes()
            .chunks(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(bytes)
            })
            .collect();

        Ok(ModelOutput::new(samples, 16000))
    }

    fn generate_sync(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<ModelOutput> {
        // For synchronous generation, we'll use the same implementation as stream
        // since llama_cpp_2 doesn't have a separate sync inference mode
        self.generate_stream(prompt, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_model_creation() -> Result<()> {
        let config = GenerationConfig::default();
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.repetition_penalty, 1.1);
        assert_eq!(config.max_length, 4096);
        assert!(config.additional_gen_config.is_empty());
        Ok(())
    }
}