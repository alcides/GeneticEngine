#!/bin/bash
echo $"#/bin/sh\nblack .\n" > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit