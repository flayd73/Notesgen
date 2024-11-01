import os
import sqlite3
import json
from typing import Dict, List
import tkinter as tk
from tkinter import filedialog, ttk
import whisper
from pyannote.audio import Pipeline
import torch
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import DATABASE_PATH, WHISPER_MODEL_SIZE, SUMMARIZER_MODEL, SIMILARITY_THRESHOLD

class LectureNotesGenerator:
    def __init__(self, auth_token: str):
        """
        Initialize the notes generator.
        auth_token: HuggingFace token for pyannote.audio access
        """
        self.database_path = DATABASE_PATH
        # Initialize Whisper
        self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        # Initialize speaker diarization
        self.diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=auth_token
        )
        self.summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)
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
                    "text": result["text"].strip()
                })
            return segments
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return []

    def format_transcript(self, segments: List[Dict]) -> str:
        """Convert segmented transcription into formatted text."""
        formatted_text = []
        current_speaker = None
        
        for segment in segments:
            if segment["speaker"] != current_speaker:
                current_speaker = segment["speaker"]
                formatted_text.append(f"\n[{current_speaker}]: ")
            formatted_text.append(segment["text"])
        
        return " ".join(formatted_text)

    def get_speaker_summary(self, segments: List[Dict]) -> Dict[str, str]:
        """Generate summaries for each speaker's contributions."""
        speaker_texts = {}
        
        # Group text by speaker
        for segment in segments:
            if segment["speaker"] not in speaker_texts:
                speaker_texts[segment["speaker"]] = []
            speaker_texts[segment["speaker"]].append(segment["text"])
        
        # Summarize each speaker's content
        summaries = {}
        for speaker, texts in speaker_texts.items():
            combined_text = " ".join(texts)
            if len(combined_text.split()) > 30:
                summary = self.summarizer(combined_text, 
                                       max_length=130, 
                                       min_length=30)[0]["summary_text"]
                summaries[speaker] = summary
            else:
                summaries[speaker] = combined_text
                
        return summaries
    
    def get_relevant_notes(self, transcript: str) -> List[Dict]:
        """Retrieve relevant existing notes based on transcript content."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT topic, content FROM notes")
        existing_notes = cursor.fetchall()
        
        # Convert transcript and existing notes to TF-IDF vectors
        documents = [transcript] + [note[1] for note in existing_notes]
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Calculate similarity between transcript and each note
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Get most relevant notes (similarity > SIMILARITY_THRESHOLD)
        relevant_notes = []
        for idx, similarity in enumerate(similarities[0]):
            if similarity > SIMILARITY_THRESHOLD:
                relevant_notes.append({
                    "topic": existing_notes[idx][0],
                    "content": existing_notes[idx][1],
                    "similarity": float(similarity)
                })
        
        conn.close()
        return sorted(relevant_notes, key=lambda x: x["similarity"], reverse=True)
    
    def generate_notes(self, audio_path: str) -> Dict:
        """Generate structured notes from audio with speaker recognition."""
        # Get transcription with speaker diarization
        segments = self.transcribe_with_speakers(audio_path)
        
        # Format full transcript
        full_transcript = self.format_transcript(segments)
        
        # Get speaker-specific summaries
        speaker_summaries = self.get_speaker_summary(segments)
        
        # Get relevant existing notes
        relevant_notes = self.get_relevant_notes(full_transcript)
        
        # Combine everything into structured notes
        notes = {
            "speaker_summaries": speaker_summaries,
            "full_transcript": full_transcript,
            "relevant_existing_notes": relevant_notes,
            "segments": segments  # Include original segments for reference
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

class LectureNotesTool(tk.Tk):
    def __init__(self, generator: LectureNotesGenerator):
        super().__init__()
        self.generator = generator
        self.title("Lecture Notes Generator")
        self.geometry("800x600")
        
        self.create_widgets()
        
    def create_widgets(self):
        # File selection
        file_frame = tk.Frame(self)
        file_frame.pack(pady=10)
        
        self.file_entry = ttk.Entry(file_frame, width=50)
        self.file_entry.pack(side=tk.LEFT)
        
        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side=tk.LEFT, padx=10)
        
        # Processing button
        process_button = ttk.Button(self, text="Generate Notes", command=self.process_file)
        process_button.pack(pady=10)
        
        # Output area
        self.output_text = tk.Text(self, height=20, width=80, font=("Arial", 12))
        self.output_text.pack(pady=10)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.flac")])
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
    
    def process_file(self):
        audio_path = self.file_entry.get()
        if audio_path:
            try:
                notes = self.generator.generate_notes(audio_path)
                self.display_notes(notes)
                self.generator.save_notes(notes, os.path.splitext(os.path.basename(audio_path))[0])
            except Exception as e:
                self.output_text.delete("1.0", tk.END)
                self.output_text.insert(tk.END, f"Error processing file: {str(e)}")
        else:
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Please select an audio file first.")
    
    def display_notes(self, notes):
        self.output_text.delete("1.0", tk.END)
        
        # Display speaker summaries
        self.output_text.insert(tk.END, "Speaker Summaries:\n")
        for speaker, summary in notes["speaker_summaries"].items():
            self.output_text.insert(tk.END, f"\n[{speaker}]:\n{summary}\n")
        
        self.output_text.insert(tk.END, "\nRelevant Existing Notes:\n")
        for note in notes["relevant_existing_notes"]:
            self.output_text.insert(tk.END, f"\nTopic: {note['topic']}\nContent: {note['content']}\nSimilarity: {note['similarity']:.2f}\n")
        
        self.output_text.insert(tk.END, f"\nFull Transcript:\n{notes['full_transcript']}")

def main():
    # Initialize the database
    init_database()
    
    # Get HuggingFace token from environment variable
    auth_token = os.getenv('HUGGINGFACE_TOKEN')
    if not auth_token:
        raise ValueError("Please set HUGGINGFACE_TOKEN environment variable")
    
    # Initialize the notes generator
    generator = LectureNotesGenerator(auth_token=auth_token)
    
    # Create and run the GUI application
    app = LectureNotesTool(generator)
    app.mainloop()

def init_database():
    """Initialize the database and create required tables."""
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
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

if __name__ == "__main__":
    main()
