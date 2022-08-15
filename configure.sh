cmake -B ./build -S ./ -DCMAKE_CXX_FLAGS="-Wfatal-errors -fsyntax-only" -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_C_COMPILER=icx
