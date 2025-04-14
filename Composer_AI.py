import streamlit as st
import pyttsx3
from pydub import AudioSegment
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs
from magenta.music import sequence_proto_to_midi_file
import note_seq

# Function to generate melody based on the genre
def generate_melody(genre):
    genre_map = {
        'pop': ('cat-mel_2bar_big', 80),
        'classical': ('hierdec-trio_16bar', 256),
        'jazz': ('hierdec-trio_16bar', 192),
        'rock': ('cat-mel_2bar_big', 64),
        'hiphop': ('cat-mel_2bar_big', 64),
        'lofi': ('cat-mel_2bar_big', 96),
        'edm': ('cat-mel_2bar_big', 64),
    }

    genre = genre.lower()
    if genre not in genre_map:
        genre = 'pop'

    model_name, length = genre_map[genre]
    config = configs.CONFIG_MAP[model_name]
    model = TrainedModel(
        config,
        batch_size=1,
        checkpoint_dir_or_path=f'https://storage.googleapis.com/magentadata/models/music_vae/{model_name}.tar'
    )

    generated = model.sample(n=1, length=length, temperature=1.0)[0]

    midi_path = "generated_melody.mid"
    wav_path = "generated_melody.wav"
    sequence_proto_to_midi_file(generated, midi_path)
    note_seq.sequence_proto_to_wav_file(generated, wav_path)

    return wav_path

# Function to synthesize lyrics to speech
def synthesize_lyrics(lyrics):
    tts_path = "lyrics.wav"
    engine = pyttsx3.init()
    engine.setProperty('rate', 130)
    engine.setProperty('volume', 1)

    voices = engine.getProperty('voices')
    tamil_voice = None
    for voice in voices:
        if 'tamil' in voice.languages[0].lower():
            tamil_voice = voice
            break
    
    if tamil_voice:
        engine.setProperty('voice', tamil_voice.id)
    engine.save_to_file(lyrics, tts_path)
    engine.runAndWait()

    return tts_path

# Function to mix music and lyrics
def mix_audio(music_path, lyrics_path, output_path="final_song.mp3"):
    music = AudioSegment.from_file(music_path)
    vocals = AudioSegment.from_file(lyrics_path)

    music = music - 6
    mixed = music.overlay(vocals)
    mixed.export(output_path, format="mp3")

    return output_path

# Streamlit App UI
st.title("AI Music Composer")
st.write("🎹 Generate music based on genre and synthesize lyrics!")

genre = st.text_input("Enter music genre (Pop, Jazz, Rock, Classical, LoFi, EDM, HipHop):")
lyrics = st.text_area("Enter your lyrics (you can use Tamil text):")

if st.button("Generate Music"):
    if genre and lyrics:
        st.write("🎼 Generating melody...")
        music_path = generate_melody(genre)
        st.write("📝 Converting lyrics to speech...")
        lyrics_path = synthesize_lyrics(lyrics)
        st.write("🎧 Mixing music and vocals...")
        final_song = mix_audio(music_path, lyrics_path)

        st.success("✅ Music generated successfully!")
        st.audio(final_song)
    else:
        st.error("Please provide both genre and lyrics!")
