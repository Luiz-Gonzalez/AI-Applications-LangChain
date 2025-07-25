from langchain_text_splitters import MarkdownHeaderTextSplitter

header_to_split_on = [
    ('#', 'Header 1'),
    ('##', 'Header 2'),
    ('###', 'Header 3')
]

md_split = MarkdownHeaderTextSplitter(
    headers_to_split_on=header_to_split_on
)

markdown_example = '''# Título do Markdown de exemplo
## Capítulo 1
Texto capítulo 1 e mais e mais texto.
Continuamos no capítulo 1!
## Capítulo 2
Texto capítulo 2 e mais e mais texto.
Continuamos no capítulo 2!
## Capítulo 3
### Seção 3.1
Texto capítulo 3 e mais e mais texto.
Continuamos no capítulo 3!
'''

splits = md_split.split_text(markdown_example)

for doc in splits:
    print(doc.page_content)
    print(doc.metadata)
    print('====================')