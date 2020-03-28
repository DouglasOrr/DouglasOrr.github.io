set -e

mkdir -p build
g++ -Wall -Wextra -Werror harsh.cpp -o build/harsh
./build/harsh

pytest nlargest_unique.py

touch build/a
python3 move.py build/a build/b
