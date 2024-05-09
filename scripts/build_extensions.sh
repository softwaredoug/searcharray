#!/bin/bash

if [[ $(uname) == "Darwin" ]]; then 
		export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
		export LDFLAGS="-L/opt/homebrew/opt/llvm/lib -Wl,-rpath,/opt/homebrew/opt/llvm/lib $LDFLAGS"
		export LDFLAGS="-L/opt/homebrew/Cellar/open-mpi/5.0.3/lib -L/opt/homebrew/opt/libevent/lib -L/opt/homebrew/opt/llvm/lib -L /opt/homebrew/Cellar/libomp/18.1.5/lib $LDFLAGS"
		export CPPFLAGS="-I/opt/homebrew/opt/llvm/include -I/opt/homebrew/Cellar/open-mpi/4.1.2/include $CPPFLAGS"
		export CFLAGS="$CPPFLAGS"
		export LIBRARY_PATH=/opt/homebrew/Cellar/open-mpi/5.0.3/lib:/opt/homebrew/opt/libevent/lib
		export CPATH=/opt/homebrew/Cellar/open-mpi/4.1.2/include
fi
echo "Building extensions..."
python setup.py build_ext --inplace
