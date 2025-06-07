# speech_stress_analyzer/utils.py

"""
Utility functions for the speech_stress_analyzer project.
This can include helper functions for data manipulation, file I/O, common calculations, etc.
"""

# Example utility function (can be removed or expanded)
def format_time(seconds: float) -> str:
    """Converts seconds to a HH:MM:SS.mmm string format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

if __name__ == "__main__":
    print("This script contains utility functions.")
    example_time = 123.456
    print(f"Example: {example_time}s is {format_time(example_time)}") 