# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <numDecks> <numHands> <numTrials> <countingThreshold>"
    exit 1
fi

mkdir -p "build"
cd "build"
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..

numDecks="$1"
numHands="$2"
numTrials="$3"
countThreshold="$4"

if ! [[ "$numDecks" =~ ^-?[0-9]+$ ]]; then
    echo "Error: The second argument must be a valid number."
    exit 1
fi

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
./build/game/simulate_counting "$numDecks" "$numHands" "$numTrials" "$countThreshold" > pnls.txt
if [ "$?" -ne 0 ]; then
    echo "Error: Failed to run the C++ binary."
    exit 1
fi

# Plot results and delete temporary file
python3 game/bots/analyze_simulate_data.py pnls.txt
rm pnls.txt
