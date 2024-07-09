mkdir -p "build"
cd "build"
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <betString> <numHands> <numTrials>"
    exit 1
fi

betString="$1"
numHands="$2"
numTrials="$3"

if ! [[ "$numHands" =~ ^-?[0-9]+$ ]]; then
    echo "Error: The second argument must be a valid number."
    exit 1
fi

if ! [[ "$numTrials" =~ ^-?[0-9]+$ ]]; then
    echo "Error: The third argument must be a valid number."
    exit 1
fi

# Run simulation
touch pnls.txt
./build/game/simulate "$betString" "$numHands" "$numTrials" > pnls.txt
if [ "$?" -ne 0 ]; then
    echo "Error: Failed to run the C++ binary."
    exit 1
fi

# Plot results and delete temporary file
python3 game/bots/histogram.py pnls.txt
rm pnls.txt

echo "Successfully wrote to output.txt"
