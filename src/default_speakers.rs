use serde_json::Value;
use std::collections::HashMap;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref DEFAULT_SPEAKERS: HashMap<String, HashMap<String, Value>> = {
        let mut speakers = HashMap::new();
        
        // English speakers
        let mut en = HashMap::new();
        en.insert("male_1".to_string(), serde_json::from_str(include_str!("default_speakers/en_male_1.json")).unwrap());
        en.insert("male_2".to_string(), serde_json::from_str(include_str!("default_speakers/en_male_2.json")).unwrap());
        en.insert("male_3".to_string(), serde_json::from_str(include_str!("default_speakers/en_male_3.json")).unwrap());
        en.insert("male_4".to_string(), serde_json::from_str(include_str!("default_speakers/en_male_4.json")).unwrap());
        en.insert("female_1".to_string(), serde_json::from_str(include_str!("default_speakers/en_female_1.json")).unwrap());
        en.insert("female_2".to_string(), serde_json::from_str(include_str!("default_speakers/en_female_2.json")).unwrap());
        speakers.insert("en".to_string(), en);
        
        // Japanese speakers
        let mut ja = HashMap::new();
        ja.insert("male_1".to_string(), serde_json::from_str(include_str!("default_speakers/ja_male_1.json")).unwrap());
        ja.insert("female_1".to_string(), serde_json::from_str(include_str!("default_speakers/ja_female_1.json")).unwrap());
        ja.insert("female_2".to_string(), serde_json::from_str(include_str!("default_speakers/ja_female_2.json")).unwrap());
        ja.insert("female_3".to_string(), serde_json::from_str(include_str!("default_speakers/ja_female_3.json")).unwrap());
        speakers.insert("ja".to_string(), ja);
        
        // Korean speakers
        let mut ko = HashMap::new();
        ko.insert("male_1".to_string(), serde_json::from_str(include_str!("default_speakers/ko_male_1.json")).unwrap());
        ko.insert("male_2".to_string(), serde_json::from_str(include_str!("default_speakers/ko_male_2.json")).unwrap());
        ko.insert("female_1".to_string(), serde_json::from_str(include_str!("default_speakers/ko_female_1.json")).unwrap());
        ko.insert("female_2".to_string(), serde_json::from_str(include_str!("default_speakers/ko_female_2.json")).unwrap());
        speakers.insert("ko".to_string(), ko);
        
        // Chinese speakers
        let mut zh = HashMap::new();
        zh.insert("male_1".to_string(), serde_json::from_str(include_str!("default_speakers/zh_male_1.json")).unwrap());
        zh.insert("female_1".to_string(), serde_json::from_str(include_str!("default_speakers/zh_female_1.json")).unwrap());
        speakers.insert("zh".to_string(), zh);
        
        speakers
    };
}