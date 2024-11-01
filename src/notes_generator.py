import os
import sqlite3
import json
from typing import Dict, List
import whisper
from pyannote.audio import Pipeline
import torch
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class LectureNotesGenerator:
    def __init__(self, auth_token: str):
        """
        Initialize the notes generator.
        auth_token: HuggingFace token for pyannote.audio access
        """
        self.database_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'notes.db')
        
        # Initialize Whisper
        self.whisper_model = whisper.load_model("small")
        
        # Initialize speaker diarization
        self.diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=auth_token
        )
        
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.vectorizer = TfidfVectorizer()
    
    def transcribe_with_speakers(self, audio_path: str) -> List[Dict]:
        """
        Transcribe audio with speaker diarization.
        Returns list of segments with speaker IDs and transcriptions.
        """
        try:
            # Perform diarization
            diarization = self.diarization(audio_path)
            
            # Extract segments with speaker information
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Transcribe this specific segment
                result = self.whisper_model.transcribe(
                    audio_path,
                    start=turn.start,
                    end=turn.end
                )
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                    "text": result["text"].strip(),
                    "duration": turn.end - turn.start
                })
            return segments
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return []

    def identify_main_lecturer(self, segments: List[Dict]) -> str:
        """
        Identify the main lecturer based on speaking time and consistency of content.
        """
        # Count total speaking time for each speaker
        speaker_duration = defaultdict(float)
        speaker_segments = defaultdict(list)
        
        for segment in segments:
            speaker_duration[segment['speaker']] += segment['duration']
            speaker_segments[segment['speaker']].append(segment['text'])
        
        # Find speaker with most speaking time
        main_lecturer = max(speaker_duration, key=speaker_duration.get)
        
        # Optional: Validate main lecturer by text cohesiveness
        def get_text_cohesiveness(texts):
            """Measure how related the texts are using TF-IDF cosine similarity"""
            if len(texts) <= 1:
                return 0
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(tfidf_matrix)
            
            # Average similarity excluding diagonal (self-similarity)
            return np.mean([
                similarities[i,j] 
                for i in range(similarities.shape[0]) 
                for j in range(similarities.shape[0]) 
                if i != j
            ])
        
        # Get text cohesiveness for potential main lecturers
        speaker_cohesiveness = {
            speaker: get_text_cohesiveness(texts)
            for speaker, texts in speaker_segments.items()
        }
        
        # Find speaker with both long speaking time and high text cohesiveness
        main_lecturer_candidates = {
            speaker: speaker_duration[speaker] * (1 + cohesiveness)
            for speaker, cohesiveness in speaker_cohesiveness.items()
        }
        
        return max(main_lecturer_candidates, key=main_lecturer_candidates.get)

    def format_transcript(self, segments: List[Dict], main_lecturer: str) -> str:
        """Convert segmented transcription into formatted text, highlighting main lecturer."""
        formatted_text = []
        
        for segment in segments:
            speaker_prefix = "[LECTURER] " if segment["speaker"] == main_lecturer else "[OTHER] "
            formatted_text.append(f"{speaker_prefix}{segment['text']}")
        
        return " ".join(formatted_text)

    def get_speaker_summary(self, segments: List[Dict], main_lecturer: str) -> Dict[str, str]:
        """Generate summaries for main lecturer and other speakers."""
        speaker_texts = defaultdict(list)
        
        # Group text by speaker
        for segment in segments:
            speaker_texts[segment["speaker"]].append(segment["text"])
        
        # Summarize contents
        summaries = {}
        for speaker, texts in speaker_texts.items():
            combined_text = " ".join(texts)
            
            # Adjust summary length based on total text
            max_length = min(max(130, len(combined_text.split()) // 3), 300)
            min_length = max(30, max_length // 3)
            
            if len(combined_text.split()) > 30:
                summary = self.summarizer(
                    combined_text, 
                    max_length=max_length, 
                    min_length=min_length
                )[0]["summary_text"]
                
                # Prefix for main lecturer
                speaker_label = "MAIN LECTURER" if speaker == main_lecturer else "OTHER SPEAKER"
                summaries[speaker_label] = summary
            else:
                summaries[speaker] = combined_text
                
        return summaries
    
    def generate_notes(self, audio_path: str) -> Dict:
        """Generate structured notes from audio with speaker recognition."""
        # Get transcription with speaker diarization
        segments = self.transcribe_with_speakers(audio_path)
        
        # Identify main lecturer
        main_lecturer = self.identify_main_lecturer(segments)
        
        # Format full transcript highlighting main lecturer
        full_transcript = self.format_transcript(segments, main_lecturer)
        
        # Get speaker-specific summaries
        speaker_summaries = self.get_speaker_summary(segments, main_lecturer)
        
        # Combine everything into structured notes
        notes = {
            "main_lecturer": main_lecturer,
            "speaker_summaries": speaker_summaries,
            "full_transcript": full_transcript,
            "segments": segments
        }
        
        return notes
    
    def save_notes(self, notes: Dict, topic: str):
        """Save generated notes to database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create a combined summary from all speakers
        combined_summary = " ".join(notes["speaker_summaries"].values())
        
        cursor.execute("""
            INSERT INTO notes (topic, content, metadata) 
            VALUES (?, ?, ?)
        """, (topic, combined_summary, json.dumps(notes)))
        
        conn.commit()
        conn.close()

def init_database(database_path):
    """Initialize the database and create required tables."""
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def main():
    # Get HuggingFace token from environment variable
    auth_token = os.getenv('HUGGINGFACE_TOKEN')
    if not auth_token:
        raise ValueError("Please set HUGGINGFACE_TOKEN environment variable")
    
    # Database path
    database_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'notes.db')
    
    # Initialize the database
    init_database(database_path)
    
    # Initialize the notes generator
    generator = LectureNotesGenerator(auth_token=auth_token)
    
    # CLI input for audio file
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate lecture notes from audio")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument("--topic", default=None, help="Topic of the lecture (optional)")
    
    args = parser.parse_args()
    
    # If no topic provided, use filename
    topic = args.topic or os.path.splitext(os.path.basename(args.audio_path))[0]
    
    try:
        # Generate notes
        notes = generator.generate_notes(args.audio_path)
        
        # Print main lecturer information
        print("\n--- Lecture Notes ---")
        print(f"Main Lecturer Identified: {notes['main_lecturer']}")
        
        # Print speaker summaries
        print("\nSpeaker Summaries:")
        for speaker, summary in notes["speaker_summaries"].items():
            print(f"\n{speaker}:")
            print(summary)
        
        # Save notes
        generator.save_notes(notes, topic)
        print(f"\nNotes saved for topic: {topic}")
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    main()
