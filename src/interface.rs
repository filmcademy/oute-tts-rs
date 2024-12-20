use anyhow::Result;
use crate::model::{GGUFModel, GenerationConfig};
use crate::prompt_processor::PromptProcessor;
use crate::audio_codec::AudioCodec;
use crate::default_speakers::DEFAULT_SPEAKERS;
use ndarray::Array;
use ndarray::IxDyn;
use crate::types::Speaker;

pub struct GGUFModelConfig {
    pub model_path: String,
    pub language: String,
    pub verbose: bool,
    pub max_seq_length: usize,
    pub n_gpu_layers: u32,
}

pub struct ModelOutput {
    audio: Vec<f32>,
    sr: u32,
}

impl ModelOutput {
    pub fn new(audio: Vec<f32>, sr: u32) -> Self {
        ModelOutput { audio, sr }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        if self.audio.is_empty() {
            eprintln!("Audio is empty, skipping save.");
            return Ok(());
        }

        // TODO: Implement audio saving logic
        // Example with hound:
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        
        for sample in &self.audio {
            // Convert f32 to i16
            let amplitude = (sample * 32767.0) as i16;
            writer.write_sample(amplitude)?;
        }
        writer.finalize()?;
        Ok(())
    }
}

pub struct InterfaceGGUF {
    config: GGUFModelConfig,
    prompt_processor: PromptProcessor,
    audio_codec: AudioCodec,
    model: GGUFModel,
}

impl InterfaceGGUF {
    pub async fn new(config: GGUFModelConfig) -> Result<Self> {
        if config.verbose {
            println!("Available speakers:");
            for (language, speakers) in DEFAULT_SPEAKERS.iter() {
                println!("Language '{}' speakers:", language);
                for (name, path) in speakers.iter() {
                    println!("  - {}: {}", name, path.as_str().unwrap_or("Invalid path"));
                }
            }
            println!();
        }

        // Initialize prompt processor with tokenizer
        let prompt_processor = PromptProcessor::new()?;

        // Initialize model
        let model = GGUFModel::new(
            &config.model_path,
            config.n_gpu_layers,
            config.max_seq_length
        )?;

        // Initialize audio codec
        let audio_codec = AudioCodec::new()?;

        Ok(InterfaceGGUF {
            config,
            prompt_processor,
            audio_codec,
            model,
        })
    }

    async fn get_audio(&self, tokens: &[i64]) -> Result<Array<f32, IxDyn>> {
        let output = self.prompt_processor.extract_audio_from_tokens(tokens);
        if output.is_empty() {
            eprintln!("No audio tokens found in the output");
            return Ok(Array::default(IxDyn(&[]))); // Return an empty array
        }

        let tensor = Array::from_shape_vec(IxDyn(&[1, output.len()]), output.to_vec())?;
        let decoded_audio = self.audio_codec.decode(tensor.as_slice().ok_or_else(|| anyhow::anyhow!("Failed to convert tensor to slice"))?)?;
        Ok(decoded_audio)
    }

    fn load_speaker(&self, path: &str) -> Result<serde_json::Value> {
        let file_content = std::fs::read_to_string(path)?;
        let speaker_data: serde_json::Value = serde_json::from_str(&file_content)?;
        Ok(speaker_data)
    }

    pub fn load_default_speaker(&self, name: &str) -> Result<serde_json::Value> {
        let name = name.to_lowercase().trim().to_string();
        let language = self.config.language.to_lowercase().trim().to_string();
        
        if self.config.verbose {
            println!("Loading speaker '{}' for language '{}'", name, language);
        }

        if !DEFAULT_SPEAKERS.contains_key(&language) {
            return Err(anyhow::anyhow!("Speaker for language {} not found. Available languages: {:?}", 
                language, 
                DEFAULT_SPEAKERS.keys().collect::<Vec<_>>()
            ));
        }

        let speakers = &DEFAULT_SPEAKERS[&language];
        if !speakers.contains_key(&name) {
            return Err(anyhow::anyhow!(
                "Speaker '{}' not found for language '{}'. Available speakers: {:?}", 
                name, 
                language,
                speakers.keys().collect::<Vec<_>>()
            ));
        }

        let speaker_path = speakers[&name].as_str().ok_or_else(|| 
            anyhow::anyhow!("Invalid speaker path format for {}", name)
        )?;

        if self.config.verbose {
            println!("Loading speaker file from: {}", speaker_path);
        }

        self.load_speaker(speaker_path)
    }

    fn check_generation_max_length(&self, max_length: Option<usize>) -> Result<()> {
        if max_length.is_none() {
            return Err(anyhow::anyhow!("max_length must be specified."));
        }
        if max_length.unwrap() > self.config.max_seq_length {
            return Err(anyhow::anyhow!(
                "Requested max_length ({}) exceeds the current max_seq_length ({})",
                max_length.unwrap(),
                self.config.max_seq_length
            ));
        }
        Ok(())
    }

    fn prepare_prompt(&self, text: &str, speaker: Option<&serde_json::Value>) -> Result<Vec<i64>> {
        let speaker = if let Some(s) = speaker {
            Some(serde_json::from_value::<Speaker>(s.clone())
                .map_err(|e| anyhow::Error::msg(e.to_string()))?)
        } else {
            None
        };
        let prompt = self.prompt_processor.get_completion_prompt(text, &self.config.language, speaker.as_ref());
        let encoded = self.prompt_processor.encode_prompt(prompt.as_str())?;
        Ok(encoded)
    }

    pub async fn generate(
        &self,
        text: &str,
        speaker: Option<&serde_json::Value>,
        temperature: Option<f32>,
        repetition_penalty: Option<f32>,
        max_length: Option<usize>,
    ) -> Result<ModelOutput> {
        let input_ids = self.prepare_prompt(text, speaker)?;
        if self.config.verbose {
            println!("Input tokens: {}", input_ids.len());
            println!("Generating audio...");
        }

        self.check_generation_max_length(max_length)?;

        let input_ids_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
        let output_i32 = self.model.generate(&input_ids_i32, &GenerationConfig {
            temperature: temperature.unwrap_or(0.1),
            max_length: max_length.unwrap_or(4096),
            repetition_penalty: repetition_penalty.unwrap_or(1.1),
        })?;
        let output: Vec<i64> = output_i32.iter().map(|&x| x as i64).collect();

        let audio = self.get_audio(&output).await?;
        if self.config.verbose {
            println!("Audio generation completed");
        }

        Ok(ModelOutput::new(audio.into_raw_vec(), self.audio_codec.get_sr()))
    }

    pub fn validate_speaker(language: &str, speaker: &str) -> Result<bool> {
        let language = language.to_lowercase().trim().to_string();
        let speaker = speaker.to_lowercase().trim().to_string();

        if !DEFAULT_SPEAKERS.contains_key(&language) {
            return Err(anyhow::anyhow!("Language {} not found", language));
        }

        let speakers = &DEFAULT_SPEAKERS[&language];
        if !speakers.contains_key(&speaker) {
            return Err(anyhow::anyhow!("Speaker {} not found for language {}", speaker, language));
        }

        Ok(true)
    }
}