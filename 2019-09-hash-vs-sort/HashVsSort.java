import java.util.Arrays;
import java.util.Collection;
import java.util.function.Function;
import java.util.HashSet;
import java.util.Random;


public class HashVsSort {
    ////////////////////////////////////////////////////////////////////////////////
    // Core

    private static int[] toIntArray(Collection<Integer> collection) {
        int[] results = new int[collection.size()];
        int index = 0;
        for (Integer value : collection) {
            results[index++] = value;
        }
        return results;
    }

    private static int[] hashUnique(int[] items) {
        HashSet<Integer> unique = new HashSet<>();
        for (int item : items) {
            unique.add(item);
        }
        return toIntArray(unique);
    }

    private static int[] sortUnique(int[] items) {
        if (items.length == 0) {
            return items;
        }
        int[] sorted = Arrays.copyOf(items, items.length);
        Arrays.sort(sorted);
        int writeIndex = 1;
        for (int i = 1; i < sorted.length; ++i) {
            if (sorted[i] != sorted[i-1]) {
                sorted[writeIndex++] = sorted[i];
            }
        }
        return Arrays.copyOf(sorted, writeIndex);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Benchmark

    private static int[] generateTestItems(int nTotal, float uniqueRatio, Random random) {
        int nUnique = (int)(nTotal * uniqueRatio);
        HashSet<Integer> unique = new HashSet<>();
        while (unique.size() < nUnique) {
            unique.add(random.nextInt());
        }
        int[] items = Arrays.copyOf(toIntArray(unique), nTotal);
        for (int i = nUnique; i < nTotal; ++i) {
            items[i] = items[random.nextInt(nUnique)];
        }
        return items;
    }

    private static void runTest(Function<int[], int[]> uniquer, String methodName, int[] items) {
        long start = System.nanoTime();
        int[] unique = uniquer.apply(items);
        long elapsed = System.nanoTime() - start;
        System.out.println(String.format("%d,%d,%s,%f", items.length, unique.length, methodName, elapsed / 1e9));
    }

    public static void main(String[] args) {
        Random random = new Random();
        System.out.println("n_total,n_unique,method,time");
        for (int rep = 0; rep < 30; ++rep) {
            for (int logTotal : Arrays.asList(16, 18, 20, 22)) {
                for (float uniqueRatio : Arrays.asList(1/16.0f, 1/2.0f, 1.0f)) {
                    int[] items = generateTestItems(1 << logTotal, uniqueRatio, random);
                    // Swap order each rep, for fairness
                    if (rep % 2 == 0) {
                        runTest(HashVsSort::hashUnique, "hash_unique", items);
                        runTest(HashVsSort::sortUnique, "sort_unique", items);
                    } else {
                        runTest(HashVsSort::sortUnique, "sort_unique", items);
                        runTest(HashVsSort::hashUnique, "hash_unique", items);
                    }
                }
            }
        }
    }
}
