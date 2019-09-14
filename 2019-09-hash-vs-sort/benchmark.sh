BUILD=build
OUT=results
CXX="g++"

set -e
set -o xtrace
mkdir -p $BUILD $OUT

echo "Profiling C++..."
$CXX hashvssort.cpp -O2 -std=c++17 -Wall -Wextra -Werror -o $BUILD/hashvssort
./$BUILD/hashvssort | tee $OUT/timings_cpp.csv

echo "Profiling Javascript..."
/opt/node/bin/node hashvssort.js | tee $OUT/timings_js.csv

echo "Profiling Java..."
javac HashVsSort.java -d build/
java -cp build/ HashVsSort | tee $OUT/timings_java.csv
