mkdir -p "build"
cd "build"
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..

timestamp=$(date +"%A, %B %d, %Y at %I:%M:%S %p")

file="LOG - $timestamp.txt"

remove_ansi() {
    sed -r "s/\x1B\[[0-9;]*[mGKH]//g"
}

touch "$file"

./build/game/player 6 | tee log.txt
cat log.txt | remove_ansi > $file
rm log.txt
