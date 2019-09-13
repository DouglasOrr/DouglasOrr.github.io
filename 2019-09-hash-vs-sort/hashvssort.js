////////////////////////////////////////////////////////////////////////////////
// Core

function hash_unique(items) {
    let unique = new Set(items);
    return Array.from(unique);
}

function sort_unique(items) {
    let sorted = items.slice().sort();
    let result = [];
    var prev;
    for (let item of sorted) {
        if (item !== prev) {
            result.push(item);
            prev = item;
        }
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Benchmarking

// Difference of two process.hrtime results, as a number of seconds (float)
function time_difference_s(t0, t1) {
    let diff_s = t1[0] - t0[0];
    let diff_ns = t1[1] - t0[1];
    return diff_s + diff_ns / 1e9;
}

function generate_test_items(n_total, unique_ratio) {
    let n_unique = Math.floor(n_total * unique_ratio);
    let unique_items = new Set();
    while (unique_items.size < n_unique) {
        unique_items.add(Math.floor(Math.random() * Math.pow(2, 32) - Math.pow(2, 31)));
    }
    let items = Array.from(unique_items);
    for (var i = 0; i < n_total - n_unique; ++i) {
        items.push(items[Math.floor(Math.random() * n_unique)]);
    }
    return items;
}

console.log("n_total,n_unique,method,time");
for (var rep = 0; rep < 30; ++rep) {
    for (let log_total of [16, 18, 20, 22]) {
        for (let unique_ratio of [1/16, 1/2, 1]) {
            let items = generate_test_items(1 << log_total, unique_ratio);
            for (let uniquer of [hash_unique, sort_unique]) {
                let start = process.hrtime();
                let unique = uniquer(items);
                let elapsed = time_difference_s(start, process.hrtime());
                console.log(`${items.length},${unique.length},${uniquer.name},${elapsed}`);
            }
        }
    }
}
