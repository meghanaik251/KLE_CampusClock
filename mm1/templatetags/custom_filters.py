import re
from django import template

register = template.Library()

@register.filter
def splitdata(meeting_time):

    """
    Custom filter to extract the time part from meeting_time string.
    For example: "W8 Wednesday 02:30 - 03:30" -> "02:30 - 03:30"
    """
    if isinstance(meeting_time, str):
        # Use regex to find the time range
        match = re.search(r'(\d{2}:\d{2} - \d{2}:\d{2})', meeting_time)
        if match:
            return match.group(0)  # Return the found time range
    return ""  # Return empty if not a string or no match found
