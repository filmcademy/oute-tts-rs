use std::collections::HashMap;
use tokenizers::Tokenizer;
use serde::{Serialize, Deserialize};

use crate::utils::number_to_words::number_to_words;

pub struct PromptProcessor {
    tokenizer: Tokenizer,
    bos: String,
    eos: String,
    special_tokens: HashMap<String, String>,
    text_prompt: String,
    map_audio_tokens: HashMap<i64, i64>,
    languages: Vec<String>,
}

impl PromptProcessor {
    pub fn new(tokenizer: Tokenizer, languages: Vec<String>) -> Self {
        let mut processor = PromptProcessor {
            tokenizer,
            bos: "<|im_start|>".to_string(),
            eos: "<|im_end|>".to_string(),
            special_tokens: HashMap::new(),
            text_prompt: "{bos}\n{text_start}{words}{text_end}\n{audio_start}\n".to_string(),
            map_audio_tokens: HashMap::new(),
            languages,
        };

        processor.special_tokens.insert("audio_code".to_string(), "<|{}|>".to_string());
        processor.special_tokens.insert("text_start".to_string(), "<|text_start|>".to_string());
        processor.special_tokens.insert("text_end".to_string(), "<|text_end|>".to_string());
        processor.special_tokens.insert("audio_start".to_string(), "<|audio_start|>".to_string());
        processor.special_tokens.insert("audio_end".to_string(), "<|audio_end|>".to_string());
        processor.special_tokens.insert("time".to_string(), "<|t_{:.2f}|>".to_string());
        processor.special_tokens.insert("code_start".to_string(), "<|code_start|>".to_string());
        processor.special_tokens.insert("code_end".to_string(), "<|code_end|>".to_string());
        processor.special_tokens.insert("text_sep".to_string(), "<|text_sep|>".to_string());

        processor.map_audio_tokens = processor.get_audio_token_map();
        processor
    }

    fn get_audio_token_map(&self) -> HashMap<i64, i64> {
        let mut map = HashMap::new();
        for i in 0..4100 {
            let token = self.tokenizer.encode(
                self.special_tokens["audio_code"].replace("{}", &i.to_string()),
                false
            ).unwrap().get_ids()[0] as i64;
            map.insert(token, i as i64);
        }
        map
    }

    pub fn process_text(&self, text: &str, language: &str) -> Vec<String> {
        if !self.languages.contains(&language.to_string()) {
            panic!("Language {} not supported, supported languages are {:?}", language, self.languages);
        }
        if language != "en" {
            panic!("Non-English languages are not supported yet.");
        }

        // Note: You'll need to implement number_to_words separately
        let text = text.to_lowercase();
        let text = regex::Regex::new(r"\d+(\.\d+)?").unwrap()
            .replace_all(&text, |caps: &regex::Captures| {
                number_to_words(&caps[0], None).unwrap_or_default()
            })
            .replace(&regex::Regex::new(r"[-_/,\.\\]").unwrap().to_string(), " ")
            .replace(&regex::Regex::new(r"[^a-z\s]").unwrap().to_string(), "");
        
        text.split(" ").map(String::from).collect()
    }

    pub fn create_audio_prompt(&self, words: &[Word]) -> String {
        words.iter()
            .map(|i| {
                let word = &i.word;
                let duration = self.special_tokens["time"]
                    .replace("{:.2f}", &format!("{:.2}", i.duration));
                let tokens = i.codes.iter()
                    .map(|c| self.special_tokens["audio_code"].replace("{}", &c.to_string()))
                    .collect::<String>();
                format!(
                    "{}{}{}{}{}",
                    word,
                    duration,
                    self.special_tokens["code_start"],
                    tokens,
                    self.special_tokens["code_end"]
                )
            })
            .collect::<Vec<String>>()
            .join("\n")
    }

    pub fn get_completion_prompt(&self, text: &str, language: &str, speaker: Option<&Speaker>) -> String {
        let mut words = self.process_text(text, language);
        
        if let Some(spk) = speaker {
            if spk.language != language {
                eprintln!("Warning: Speaker language {} does not match text language {}", spk.language, language);
            }
            let mut speaker_words = self.process_text(&spk.text, &spk.language);
            words.append(&mut speaker_words);
        }

        let words = words.iter()
            .map(|word| word.trim().to_string())
            .collect::<Vec<String>>()
            .join(&self.special_tokens["text_sep"]);

        let mut prompt = self.text_prompt
            .replace("{bos}", &self.bos)
            .replace("{text_start}", &self.special_tokens["text_start"])
            .replace("{words}", &words)
            .replace("{text_end}", &self.special_tokens["text_end"])
            .replace("{audio_start}", &self.special_tokens["audio_start"]);

        if let Some(spk) = speaker {
            prompt.push_str(&self.create_audio_prompt(&spk.words));
        }

        prompt
    }

    pub fn extract_audio_from_tokens(&self, tokens: &[i64]) -> Vec<i64> {
        let mut result = Vec::new();
        for token in tokens {
            if let Some(&x) = self.map_audio_tokens.get(token) {
                result.push(x);
            }
        }
        result
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Speaker {
    pub language: String,
    pub text: String,
    pub words: Vec<Word>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Word {
    pub word: String,
    pub duration: f64,
    pub codes: Vec<i32>,
}
