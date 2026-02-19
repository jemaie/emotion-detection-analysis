try:
    import moviepy.editor as mp
    print("moviepy available")
except ImportError:
    print("moviepy not found")

try:
    import librosa
    print("librosa available")
except ImportError:
    print("librosa not found")

try:
    import pydub
    print("pydub available")
except ImportError:
    print("pydub not found")
