use std::path::Path;
use std::sync::Arc;
use ort::{Environment, SessionBuilder, Value, Session};
use ndarray::{Array, CowArray, IxDyn};
use anyhow::{Result, Context};

pub struct AudioCodec {
    session: Session,
    pub sr: u32,
}

impl AudioCodec {
    pub fn new() -> Result<Self> {
        let models_dir = "models";
        let model_path = Path::new(models_dir).join("decoder.onnx");

        if !model_path.exists() {
            anyhow::bail!(
                "ONNX model not found at {}. Ensure the project was built correctly.", 
                model_path.display()
            );
        }

        // Initialize environment with ONNX Runtime
        let environment = Environment::builder()
            .with_name("wavtokenizer_environment")
            .build()
            .context("Failed to initialize ONNX Runtime environment")?;

        // Load model and create session
        let environment_arc = Arc::new(environment);
        let session = SessionBuilder::new(&environment_arc)?
            .with_model_from_file(&model_path)
            .context("Failed to load ONNX model")?;

        Ok(AudioCodec {
            session,
            sr: 24000,
        })
    }

    pub fn decode(&self, codes: &[i64]) -> Result<Array<f32, IxDyn>> {
        // Create input tensor with shape [1, codes.length]
        let shape = [1, codes.len()];
        let array = Array::from_shape_vec(IxDyn(&shape), codes.to_vec())
            .context("Failed to create input array")?;
        
        // Convert to CowArray for ONNX Runtime
        let cow_array = CowArray::from(array);

        // Create input Value for ONNX Runtime
        let input_tensor = Value::from_array(self.session.allocator(), &cow_array)
            .context("Failed to create input tensor")?;

        // Run inference with named inputs matching the ONNX model
        let input_tensors = vec![input_tensor];
        let outputs = self.session.run(input_tensors)
            .context("Failed to run model inference")?;

        // Extract waveform from outputs
        let waveform = outputs[0].try_extract::<f32>()
            .context("Failed to extract output waveform")?;
        
        Ok(waveform.view().to_owned())
    }

    pub fn sample_rate(&self) -> i32 {
        self.sr as i32
    }

    pub fn get_sr(&self) -> u32 {
        self.sr
    }
}