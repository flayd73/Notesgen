import os
from src.notes_generator import LectureNotesGenerator
from src.database import init_database

def main():
    # Initialize the database
    init_database()
    
    # Get HuggingFace token from environment variable
    auth_token = os.getenv('HUGGINGFACE_TOKEN')
    if not auth_token:
        raise ValueError("Please set HUGGINGFACE_TOKEN environment variable")
    
    # Initialize the notes generator
    generator = LectureNotesGenerator(auth_token=auth_token)
    
    # Process an audio file
    audio_path = "path_to_your_audio_file.wav"  # Replace with your audio file
    try:
        notes = generator.generate_notes(audio_path)
        
        # Print speaker summaries
        print("\nSpeaker Summaries:")
        for speaker, summary in notes["speaker_summaries"].items():
            print(f"\n{speaker}:")
            print(summary)
        
        # Save the notes
        generator.save_notes(notes, "Example Topic")
        print("\nNotes saved successfully!")
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    main()
