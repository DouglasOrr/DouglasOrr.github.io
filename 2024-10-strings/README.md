# Strings

Play around with and benchmark `std::string`, `std::string_view`, `std::reference_wrapper<std::string>`.

My benchmark results on [data/2024-10-strings](https://github.com/DouglasOrr/DouglasOrr.github.io/tree/data/2024-10-strings).

To run manually:

```sh
clang++ -Wall -Wextra -Werror -O3 -std=c++20 -stdlib=libc++ demo.cpp -o demo
./demo
```

To run the benchmarks (results in `out/benchmarks.jsonl`):

```sh
./run_benchmarks.sh
```
