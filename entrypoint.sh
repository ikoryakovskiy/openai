#!/bin/bash
set -e

cp /grl/qt-build/py_env.* /openai
exec "$@"
