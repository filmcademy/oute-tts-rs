use std::fs;
use std::path::Path;
use std::io::Write;
use reqwest;

fn download_file(url: &str, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Skip download if RUST_ANALYZER is set
    if std::env::var("RUST_ANALYZER").is_ok() {
        println!("cargo:warning=Skipping download in rust-analyzer context");
        return Ok(());
    }

    println!("cargo:warning=Downloading from {}...", url);
    
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))  // 10 minute timeout
        .build()?;
    
    let response = client.get(url)
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        .send()?;
    
    if !response.status().is_success() {
        return Err(format!("Failed to download file from {} - Status: {}", url, response.status()).into());
    }
    
    let content = response.bytes()?;
    if String::from_utf8_lossy(&content).starts_with("version https://git-lfs.github.com/spec/v1") {
        return Err("Received Git LFS pointer instead of actual file".into());
    }

    // Create parent directories if they don't exist
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::File::create(path)?;
    file.write_all(&content)?;
    
    println!("cargo:warning=Successfully downloaded to {}", path);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models_dir = "models";
    fs::create_dir_all(models_dir)?;

    // Download tokenizer if it doesn't exist
    let tokenizer_path = format!("{}/tokenizer.json", models_dir);
    if !Path::new(&tokenizer_path).exists() {
        println!("cargo:warning=Downloading tokenizer...");
        download_file(
            "https://huggingface.co/onnx-community/OuteTTS-0.2-500M/resolve/main/tokenizer.json",
            &tokenizer_path
        )?;
    }

    // Download GGUF model if it doesn't exist
    let gguf_path = format!("{}/OuteTTS-0.2-500M-FP16.gguf", models_dir);
    if !Path::new(&gguf_path).exists() {
        println!("cargo:warning=Downloading GGUF model (this may take a while)...");
        download_file(
            "https://huggingface.co/OuteAI/OuteTTS-0.2-500M-GGUF/resolve/main/OuteTTS-0.2-500M-FP16.gguf",
            &gguf_path
        )?;
    }

    // Download ONNX model if it doesn't exist
    let model_path = format!("{}/decoder.onnx", models_dir);
    if !Path::new(&model_path).exists() {
        println!("cargo:warning=Downloading ONNX decoder model...");
        download_file(
            "https://huggingface.co/onnx-community/WavTokenizer-large-speech-75token_decode/resolve/main/onnx/model.onnx",
            &model_path
        )?;
    }

    // Tell Cargo to rerun this script if the model files are deleted
    println!("cargo:rerun-if-changed={}", tokenizer_path);
    println!("cargo:rerun-if-changed={}", gguf_path);
    println!("cargo:rerun-if-changed={}", model_path);

    Ok(())
}