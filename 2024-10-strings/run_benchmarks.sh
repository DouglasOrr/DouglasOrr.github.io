mkdir -p build/ out/
set -e

flags="-Wall -Wextra -Werror -O3 -std=c++20"

run_test () {
    $1 $flags $2 demo.cpp -o build/demo
    ./build/demo "out/benchmarks.jsonl"
}

reps=4
for i in $(seq 1 $reps);
do
    run_test "g++"
    run_test "clang++" "-stdlib=libc++"
    run_test "clang++" "-stdlib=libstdc++"
done
