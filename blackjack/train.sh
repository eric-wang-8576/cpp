mkdir -p "build"
cd "build"
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..

./build/game/book_training 6
