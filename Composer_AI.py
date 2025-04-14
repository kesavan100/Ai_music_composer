import streamlit as st
import pyttsx3
from pydub import AudioSegment
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs
from magenta.music import sequence_proto_to_midi_file
import note_seq
import tempfile

# Generate music based on genre
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
        st.warning(f"⚠️ Genre '{genre}' not found. Using default: Pop.")
        genre = 'pop'

    model_name, length = genre_map[genre]
    st.write(f"🎵 Loading model for genre: {genre.title()}")

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
    st.write(f"🎶 Music generated and saved to: {wav_path}")

    return wav_path

# Convert lyrics to speech (including Tamil support)
def synthesize_lyrics(lyrics):
    tts_path = "lyrics.wav"
    st.write("🗣️ Converting lyrics to speech...")

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
        st.write(f"🎤 Using Tamil voice: {tamil_voice.name}")
    else:
        st.warning("⚠️ No Tamil voice found. Using default voice.")

    engine.save_to_file(lyrics, tts_path)
    engine.runAndWait()
    st.write(f"✅ Lyrics saved as audio: {tts_path}")

    return tts_path

# Mix music and voice
def mix_audio(music_path, lyrics_path, output_path="final_song.mp3"):
    st.write("🎧 Mixing vocals and music...")

    music = AudioSegment.from_file(music_path)
    vocals = AudioSegment.from_file(lyrics_path)

    music = music - 6
    mixed = music.overlay(vocals)

    mixed.export(output_path, format="mp3")
    st.write(f"🎼 Final song saved as: {output_path}")

    return output_path

# Streamlit Interface
st.title("AI Music Composer")
st.header("Generate Music with Your Lyrics")

genre = st.selectbox("Choose Genre", ['Pop', 'Classical', 'Jazz', 'Rock', 'HipHop', 'LoFi', 'EDM'])
lyrics = st.text_area("Enter Lyrics (can include Tamil text)")

if st.button("Generate Music"):
    if genre and lyrics:
        st.write("🎼 Generating music and vocals...")
        music_path = generate_melody(genre)
        lyrics_path = synthesize_lyrics(lyrics)
        final_output_path = mix_audio(music_path, lyrics_path)

        st.audio(final_output_path, format="audio/mp3")
    else:
        st.error("Please enter both genre and lyrics.")

