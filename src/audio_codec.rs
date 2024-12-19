use tch::Tensor;

pub struct AudioCodec {
    wavtokenizer: Box<dyn Fn(Tensor) -> Tensor>,
    sr: i32,
}

impl AudioCodec {
    pub fn new<F>(wavtokenizer: F) -> Self 
    where
        F: Fn(Tensor) -> Tensor + 'static,
    {
        AudioCodec {
            wavtokenizer: Box::new(wavtokenizer),
            sr: 24000,
        }
    }

    pub async fn decode(&self, codes: &[i64]) -> Tensor {
        // Create tensor with shape [1, codes.length]
        let codes_tensor: Tensor = Tensor::from_slice(codes)
            .view([1, codes.len() as i64]);

        // Pass tensor through wavtokenizer function
        (self.wavtokenizer)(codes_tensor)
    }
}