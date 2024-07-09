# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <numHands> <numTrials> <countingThreshold>"
    exit 1
fi

mkdir -p "build"
cd "build"
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..


numHands="$1"
numTrials="$2"
countThreshold="$3"

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
./build/game/simulate_counting "$numHands" "$numTrials" "$countThreshold" > pnls.txt
if [ "$?" -ne 0 ]; then
    echo "Error: Failed to run the C++ binary."
    exit 1
fi

# Plot results and delete temporary file
python3 game/bots/analyze_simulate_data.py pnls.txt
rm pnls.txt
