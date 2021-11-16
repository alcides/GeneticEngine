import os




INSERTCODE = "<insertCodeHere>"
CWD = os.path.dirname(os.path.realpath(__file__))

bla = "print(1 + 1)"


# ====================
# -- Grammar
# ====================




# ====================
# -- Combining
# ====================

def embed_code_in_container(container_footer: str,container_header: str, code: str, indent=""):
    return ()
    

def get_container_code(container_file: str):
    with open(os.path.join(CWD, container_file), 'r') as container:
            container_code = container.read()
    embedding_location = container_code.index(INSERTCODE)
    container_header, container_footer = "", ""
    if embedding_location > 0:
        container_footer = container_code[:embedding_location]
        container_header = container_code[embedding_location + len(INSERTCODE):]
    else:
        container_header = container_code[len(INSERTCODE):]

    return container_footer,container_header

container_footer,container_header = get_container_code("Compare String Lengths-Embed.txt")

print(container_footer)

print("\n----------------\n----------------\n----------------")

print(container_header)
