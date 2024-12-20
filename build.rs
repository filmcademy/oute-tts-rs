use std::fs;
use std::path::Path;
use std::io::Write;
use reqwest;

fn download_file(url: &str, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let response = reqwest::blocking::get(url)?;
    
    if !response.status().is_success() {
        panic!("Failed to download file from {}", url);
    }
    
    let content = response.bytes()?;
    if String::from_utf8_lossy(&content).starts_with("version https://git-lfs.github.com/spec/v1") {
        panic!("Received Git LFS pointer instead of actual file");
    }

    let mut file = fs::File::create(path)?;
    file.write_all(&content)?;
    
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
    println!("cargo:rerun-if-changed={}", model_path);

    Ok(())
}