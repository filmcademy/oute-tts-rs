mod utils;
use utils::number_to_words::number_to_words;

mod prompt_processor;
mod audio_codec;

fn main() {
    let number = "1234567890";
    let words = number_to_words(number, None);
    println!("{:?}", words);
}
