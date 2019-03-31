__all__ = [
    'tokenize_dsl_code',
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
