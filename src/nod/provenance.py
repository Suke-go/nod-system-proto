"""Identify the installed implementation independently of a Git checkout."""
import hashlib
from importlib.resources import files


def source_digest():
    def walk(root,prefix=''):
        for node in root.iterdir():
            name=prefix+node.name
            if node.is_dir() and node.name!='__pycache__':
                yield from walk(node,name+'/')
            elif node.is_file() and node.name.endswith(('.py','.json','.html')):
                yield name,node
    digest=hashlib.sha256()
    for name,node in sorted(walk(files('nod')),key=lambda item:item[0]):
        digest.update(name.encode('utf-8')+b'\0'+node.read_bytes()+b'\0')
    return digest.hexdigest()
