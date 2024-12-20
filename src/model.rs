use anyhow::Result;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::{LlamaSampler, params::LlamaSamplerChainParams};
use llama_cpp_2::token::LlamaToken;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex, Once};
use lazy_static::lazy_static;

static INIT: Once = Once::new();

lazy_static! {
    static ref BACKEND: Arc<Mutex<Option<Arc<LlamaBackend>>>> = Arc::new(Mutex::new(None));
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub max_length: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            repetition_penalty: 1.1,
            max_length: 4096,
        }
    }
}

pub struct GGUFModel {
    model: Arc<LlamaModel>,
    context: Arc<Mutex<llama_cpp_2::context::LlamaContext<'static>>>,
    backend: Arc<LlamaBackend>,
}

impl GGUFModel {
    pub fn new(
        model_path: impl AsRef<Path>,
        n_gpu_layers: u32,
        max_seq_length: usize,
    ) -> Result<Self> {
        let backend = {
            let mut backend_guard = BACKEND.lock().unwrap();
            if let Some(ref backend) = *backend_guard {
                backend.clone()
            } else {
                let new_backend = Arc::new(LlamaBackend::init()?);
                *backend_guard = Some(new_backend.clone());
                INIT.call_once(|| {});
                new_backend
            }
        };

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers);
        
        let model = Arc::new(LlamaModel::load_from_file(&backend, model_path, &model_params)?);
        
        let ctx_size = NonZeroU32::new(max_seq_length as u32)
            .ok_or_else(|| anyhow::anyhow!("Context size must be greater than zero"))?;
        
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size));
            
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
        input_tokens: &[i32],
        config: &GenerationConfig,
    ) -> Result<Vec<i32>> {
        let mut tokens = Vec::new();
        let context = self.context.lock().unwrap();
        
        let sampler = LlamaSampler::new(LlamaSamplerChainParams::default())?
            .add_temp(config.temperature)
            .add_penalties(
                self.model.n_vocab() as i32,
                0,
                0,
                0,
                config.repetition_penalty,
                1.0,
                1.0,
                false,
                false
            );

        tokens.extend_from_slice(input_tokens);

        while tokens.len() < config.max_length {
            let token = sampler.sample(&context, tokens.len() as i32);
            tokens.push(token.0);
            
            if self.model.is_eog_token(LlamaToken(token.0)) {
                break;
            }
        }

        Ok(tokens)
    }
}