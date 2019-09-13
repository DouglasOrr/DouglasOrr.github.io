BUILD=build
OUT=results
CXX="g++"

set -e
set -o xtrace
mkdir -p $BUILD $OUT

$CXX hashvssort.cpp -O2 -std=c++17 -Wall -Wextra -Werror -o $BUILD/hashvssort
./$BUILD/hashvssort | tee $OUT/timings_cpp.csv

/opt/node/bin/node hashvssort.js | tee $OUT/timings_js.csv
