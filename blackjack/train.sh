mkdir -p "build"
cd "build"
cmake ..
make
cd ..

./build/game/book_training 6
