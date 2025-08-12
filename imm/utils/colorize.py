"""
Terminal text colorization utility.

Converted from TensorFlow implementation.
"""

from typing import Optional


def colorize(text: str, color: str = 'white', bold: bool = False) -> str:
    """
    Colorize text for terminal output.
    
    Args:
        text: Text to colorize
        color: Color name ('red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white')
        bold: Whether to make text bold
        
    Returns:
        Colorized text string
    """
    colors = {
        'red': '\033[91m',
        'green': '\033[92m', 
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    
    # Start with color code
    result = colors.get(color.lower(), colors['white'])
    
    # Add bold if requested
    if bold:
        result = '\033[1m' + result
    
    # Add text and reset
    result += text + colors['reset']
    
    return result
