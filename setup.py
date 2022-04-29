from __future__ import annotations

import setuptools

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "Readme.md").read_text()


setuptools.setup(
    long_description=long_description,
    long_description_content_type='text/markdown'
)
