mod model;
mod prompt_processor;
mod default_speakers;
mod utils;
mod audio_codec;

use clap::Parser;
use anyhow::Result;
use crate::model::{GGUFModel, GenerationConfig, ModelOutput};
use crate::prompt_processor::PromptProcessor;
use crate::default_speakers::DEFAULT_SPEAKERS;
use audio_codec::AudioCodec;
use tokio;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(long)]
    model: String,

    /// Text to synthesize
    #[arg(long)]
    text: String,

    /// Language for synthesis
    #[arg(long, default_value = "en")]
    language: String,

    /// Speaker voice to use
    #[arg(long, default_value = "male_1")]
    speaker: String,

    /// Output audio file path
    #[arg(long, default_value = "output.wav")]
    output: String,

    /// Number of GPU layers to use
    #[arg(long, default_value_t = 0)]
    gpu_layers: u32,

    /// Generation temperature
    #[arg(long, default_value_t = 0.1)]
    temperature: f32,
}

impl GGUFModel {
    pub fn str_to_token(&self, text: &str, add_bos: bool) -> Result<Vec<i64>> {
        unimplemented!("Need to implement str_to_token")
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Validate language
    if !["en", "ja", "ko", "zh"].contains(&args.language.as_str()) {
        anyhow::bail!("Unsupported language. Must be one of: en, ja, ko, zh");
    }

    // Initialize model
    let model = GGUFModel::new(
        &args.model,
        args.gpu_layers,
        4096,
        None,
    )?;

    // Get speaker data
    let speakers = DEFAULT_SPEAKERS.get(&args.language)
        .ok_or_else(|| anyhow::anyhow!("Language not found"))?;
    
    let speaker_data = speakers.get(&args.speaker)
        .ok_or_else(|| anyhow::anyhow!("Speaker not found"))?;

    // Create generation config
    let generation_config = GenerationConfig {
        temperature: args.temperature,
        repetition_penalty: 1.1,
        max_length: 4096,
        additional_gen_config: std::collections::HashMap::new(),
    };

    // Generate prompt
    let prompt = format!("<|{}|>{}", args.speaker, args.text);

    // Generate audio
    let output = model.generate(&prompt, &generation_config, false)?;
    
    // Save to file
    output.save(&args.output)?;
    
    println!("Audio saved to: {}", args.output);
    
    Ok(())
}
