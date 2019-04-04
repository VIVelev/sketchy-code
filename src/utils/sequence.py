__all__ = [
    'tokenize_dsl_code',
    'tokenize_html_code',
]


def tokenize_dsl_code(code):
    tokens = code.split()
    final_tokens = []

    for token in tokens:
        if ',' in token:
            final_tokens.append(token.split(',')[0])
            final_tokens.append(',')
        else:
            final_tokens.append(token)

    return final_tokens

def tokenize_html_code(code):
    html_tags = []
    closing_bound = 0
    
    while True:
        closing_idx = code.find('>', closing_bound)
        if closing_idx == -1:
            break

        tag = code[closing_bound:closing_idx+1]
        tag = tag[tag.find('<'):tag.find('>')+1]

        html_tags.append(tag)
        closing_bound = closing_idx+1

    return html_tags
