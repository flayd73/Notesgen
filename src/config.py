import os

# Database configuration
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'notes.db')

# Model configurations
WHISPER_MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large
SUMMARIZER_MODEL = "facebook/bart-large-cnn"

# Similarity threshold for relevant notes
SIMILARITY_THRESHOLD = 0.3

