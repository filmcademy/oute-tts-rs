use serde::Deserialize;
use crate::prompt_processor::Word;

#[derive(Deserialize)]
pub struct Speaker {
    pub name: String,
    pub language: String,
    pub text: String,
    pub words: Vec<Word>,
} 