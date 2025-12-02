# text_to_image.py - Generate an image with text, auto-wrapped to fit
#
# Usage:
#   python text_to_image.py                           # Default text
#   python text_to_image.py "Your text here"          # Custom text
#   python text_to_image.py Your text here            # Unquoted also works
#
# The text is automatically wrapped and font size adjusted to fit the image.
#
# Requires: matplotlib

import sys
import textwrap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Get text from command line or use default
text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello, world!"

# Replace \n with actual newlines
text = text.replace("\\n", "\n")

# Image dimensions
width, height = 4000, 1500
aspect = width / height

def fit_text(text, target_width_chars=40):
    """Wrap text and find optimal font size to fit in image."""
    # Start with reasonable wrap width based on text length
    if len(text) < target_width_chars:
        wrapped = text
    else:
        wrapped = "\n".join(textwrap.wrap(text, width=target_width_chars))

    lines = wrapped.split("\n")
    num_lines = len(lines)
    max_line_len = max(len(line) for line in lines)

    # Calculate font size to fit
    # Approximate: font size inversely proportional to max dimension needed
    font_size_by_width = 600 * 15 / max(max_line_len, 1)
    font_size_by_height = 600 / max(num_lines, 1)
    font_size = min(font_size_by_width, font_size_by_height, 600)

    return wrapped, font_size

wrapped_text, fontsize = fit_text(text)

fig, ax = plt.subplots(figsize=(40, 15))
ax.text(0.5, 0.5, wrapped_text, fontsize=fontsize,
        ha='center', va='center', wrap=False,
        fontfamily='monospace')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.savefig('text_to_image.png', dpi=100, bbox_inches='tight', pad_inches=0.1)
print(f'Saved text_to_image.png with text:\n"{text}"')
print(f'Font size: {fontsize:.0f}, Lines: {len(wrapped_text.split(chr(10)))}')
