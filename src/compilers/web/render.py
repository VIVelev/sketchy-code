__all__ = [
    'TEXT_PLACEHOLDER',
    'render_content_with_text',
]


TEXT_PLACEHOLDER = '[]'

def render_content_with_text(key, value):
    return value.replace(TEXT_PLACEHOLDER, 'Some text here')
