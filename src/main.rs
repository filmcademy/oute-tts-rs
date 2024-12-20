mod model;
mod prompt_processor;
mod default_speakers;
mod utils;
mod audio_codec;
mod interface;
mod types;

use clap::Parser;
use anyhow::Result;
use interface::{InterfaceGGUF, GGUFModelConfig};

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

    /// Maximum sequence length
    #[arg(long, default_value_t = 4096)]
    max_length: usize,

    /// Enable verbose output
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Repetition penalty
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Validate language
    if !["en", "ja", "ko", "zh"].contains(&args.language.as_str()) {
        anyhow::bail!("Unsupported language. Must be one of: en, ja, ko, zh");
    }

    // Create model config
    let config = GGUFModelConfig {
        model_path: args.model,
        language: args.language,
        verbose: args.verbose,
        n_gpu_layers: args.gpu_layers,
        max_seq_length: args.max_length,
    };

    // First validate that the speaker exists
    if config.verbose {
        println!("Validating speaker configuration...");
    }
    
    // Pre-validate speaker before model initialization
    InterfaceGGUF::validate_speaker(&config.language, &args.speaker)?;

    // Initialize interface (including model) only after speaker validation
    if config.verbose {
        println!("Initializing model...");
    }
    let interface = InterfaceGGUF::new(config).await?;

    // Load speaker after validation
    let speaker = interface.load_default_speaker(&args.speaker)?;

    let output = interface.generate(
        &args.text,
        Some(&speaker),
        Some(args.temperature),
        Some(args.repetition_penalty),
        Some(args.max_length),
    ).await?;

    // Save to file
    output.save(&args.output)?;
    
    if args.verbose {
        println!("Audio saved to: {}", args.output);
    }
    
    Ok(())
}
