# This document explains how to use Nuitka to compile Genetic Engine

Note that pyenv macos Python does not work. You need to use homebrew's version.

```
/opt/homebrew/bin/python3 -m pip install nuitka
/opt/homebrew/bin/python3 -m pip install -r requirements.txt

/opt/homebrew/bin/python3 -m nuitka --standalone  --follow-imports --enable-plugin=numpy examples/game_of_life.py
Z3_LIBRARY_PATH=/opt/homebrew/Cellar/z3/4.12.1/lib/ game_of_life.dist/game_of_life.bin
```
