__all__ = [
    'Node',
]

class Node:
    def __init__(self, key, parent, content_holder):
        self.key = key
        self.parent = parent
        self.children = []
        self.content_holder = content_holder

    def add_child(self, child):
        self.children.append(child)

    def show(self):
        print(self.key)
        for child in self.children:
            child.show()

    def render(self, mapping, render_function=None):
        content = ''
        for child in self.children:
            content += child.render(mapping, render_function)

        value = mapping[self.key]
        if render_function is not None:
            value = render_function(self.key, value)

        if len(self.children) > 0:
            value = value.replace(self.content_holder, content)

        return value
