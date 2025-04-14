import streamlit as st
import os
from pydub import AudioSegment
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs
from magenta.music import sequence_proto_to_midi_file
import note_seq
from gtts import gTTS

# Function to generate music based on genre
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
    model = TrainedModel(config, batch_size=1, checkpoint_dir_or_path=f'https://storage.googleapis.com/magentadata/models/music_vae/{model_name}.tar')

    generated = model.sample(n=1, length=length, temperature=1.0)[0]

    midi_path = "generated_melody.mid"
    wav_path = "generated_melody.wav"
    sequence_proto_to_midi_file(generated, midi_path)
    note_seq.sequence_proto_to_wav_file(generated, wav_path)

    return wav_path

# Convert lyrics to speech
def synthesize_lyrics(lyrics):
    tts_path = "lyrics.mp3"

    tts = gTTS(text=lyrics, lang='ta')
    tts.save(tts_path)

    return tts_path

# Mix music and voice
def mix_audio(music_path, lyrics_path, output_path="final_song.mp3"):
    music = AudioSegment.from_file(music_path)
    vocals = AudioSegment.from_file(lyrics_path)

    music = music - 6
    mixed = music.overlay(vocals)

    mixed.export(output_path, format="mp3")
    return output_path

# Streamlit app
def main():
    st.title('AI Music Composer')

    genre = st.text_input('Enter music genre (Pop, Jazz, Rock, Classical, LoFi, EDM, HipHop):', 'pop')
    lyrics = st.text_area('Enter your lyrics (Tamil supported):', 'உங்கள் பாடல் இங்கே இடுக.')

    if st.button('Generate Song'):
        st.write("🎶 Generating Music and Lyrics...")

        # Generate music based on genre
        music_path = generate_melody(genre)
        st.write("🎵 Music Generated!")

        # Convert lyrics to speech
        lyrics_path = synthesize_lyrics(lyrics)
        st.write("🗣️ Lyrics Synthesized!")

        # Mix music and vocals
        final_song_path = mix_audio(music_path, lyrics_path)
        st.write(f"🎼 Final song generated: {final_song_path}")

        # Display the audio player
        st.audio(final_song_path, format='audio/mp3')

if __name__ == "__main__":
    main()
