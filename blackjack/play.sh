mkdir -p "build"
cd "build"
cmake ..
make
cd ..

timestamp=$(date +"%A, %B %d, %Y at %I:%M:%S %p")

file="LOG - $timestamp.txt"

touch "$file"

./build/player 6 | tee tmp.txt
cat tmp.txt | sed 's/\x1b\[[0-9;]*m//g' > $file
rm tmp.txt
