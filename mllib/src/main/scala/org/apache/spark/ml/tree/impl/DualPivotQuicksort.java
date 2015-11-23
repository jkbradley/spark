package org.apache.spark.ml.tree.impl;

/**
 * Created by fabuzaid21 on 11/2/15.
 * Helper utility class for sorting
 * Double arrays with corresponding Int indices in
 * {@link org.apache.spark.ml.tree.impl.AltDT}.
 * Provides a more efficient alternative to:
 *
 * <pre>
 * {@code
 * val doubles = Array.fill[Double](len)(Random.nextDouble)
 * val idxs = Array.fill[Int](len)(Random.nextInt(len))
 * val (sortedDoubles, sortedIdxs) = doubles.zip(idxs).sorted.unzip
 * }
 * </pre>
 *
 * NOTE: This implementation is directly borrowed from
 * Yaroslavskiy et al.'s implementation of the Dual-Pivot
 * Quicksort Algorithm in OpenJDK 7, which can be found
 * <a href="http://grepcode.com/file/repository.grepcode.
 * com/java/root/jdk/openjdk/7-b147/java/util/DualPivotQuicksort.
 * java">here</a>.
 */
public class DualPivotQuicksort {

    /**
     * Prevents instantiation.
     */
    private DualPivotQuicksort() {}

    /**
     * The maximum number of runs in merge sort.
     */
    private static final int MAX_RUN_COUNT = 67;

    /**
     * The maximum length of run in merge sort.
     */
    private static final int MAX_RUN_LENGTH = 33;

    /**
     * If the length of an array to be sorted is less than this
     * constant, Quicksort is used in preference to merge sort.
     */
    private static final int QUICKSORT_THRESHOLD = 286;

    /**
     * If the length of an array to be sorted is less than this
     * constant, insertion sort is used in preference to Quicksort.
     */
    private static final int INSERTION_SORT_THRESHOLD = 47;

    /**
     * Sorts the specified array.
     *
     * @param arr the array to be sorted
     * @param idxs the corresponding indices array, updated to match arr's sorted order
     */
    public static void sort(double[] arr, int[] idxs) {
        sort(arr, idxs, 0, arr.length - 1);
    }

    /**
     * Sorts the specified range of the array.
     *
     * @param arr the array to be sorted
     * @param idxs the corresponding indices array, updated to match arr's sorted order
     * @param left the index of the first element, inclusive, to be sorted
     * @param right the index of the last element, inclusive, to be sorted
     */
    public static void sort(double[] arr, int[] idxs, int left, int right) {
        /*
         * Phase 1: Move NaNs to the end of the array.
         */
        while (left <= right && Double.isNaN(arr[right])) {
            --right;
        }
        for (int k = right; --k >= left; ) {
            double ak = arr[k];
            int bk = idxs[k];
            if (ak != ak) { // arr[k] is NaN
                arr[k] = arr[right];
                arr[right] = ak;
                idxs[k] = idxs[right];
                idxs[right] = bk;
                --right;
            }
        }

        /*
         * Phase 2: Sort everything except NaNs (which are already in place).
         */
        doSort(arr, idxs, left, right);

        /*
         * Phase 3: Place negative zeros before positive zeros.
         */
        int hi = right;

        /*
         * Find the first zero, or first positive, or last negative element.
         */
        while (left < hi) {
            int middle = (left + hi) >>> 1;
            double middleValue = arr[middle];

            if (middleValue < 0.0d) {
                left = middle + 1;
            } else {
                hi = middle;
            }
        }

        /*
         * Skip the last negative value (if any) or all leading negative zeros.
         */
        while (left <= right && Double.doubleToRawLongBits(arr[left]) < 0) {
            ++left;
        }

        /*
         * Move negative zeros to the beginning of the sub-range.
         *
         * Partitioning:
         *
         * +----------------------------------------------------+
         * |   < 0.0   |   -0.0   |   0.0   |   ?  ( >= 0.0 )   |
         * +----------------------------------------------------+
         *              ^          ^         ^
         *              |          |         |
         *             left        p         k
         *
         * Invariants:
         *
         *   all in (*,  left)  <  0.0
         *   all in [left,  p) == -0.0
         *   all in [p,     k) ==  0.0
         *   all in [k, right] >=  0.0
         *
         * Pointer k is the first index of ?-part.
         */
        for (int k = left, p = left - 1; ++k <= right; ) {
            double ak = arr[k];
            if (ak != 0.0d) {
                break;
            }
            if (Double.doubleToRawLongBits(ak) < 0) { // ak is -0.0d
                arr[k] = 0.0d;
                arr[++p] = -0.0d;
            }
        }
    }

    /**
     * Sorts the specified range of the array.
     *
     * @param arr the array to be sorted
     * @param idxs the corresponding indices array, updated to match arr's sorted order
     * @param left the index of the first element, inclusive, to be sorted
     * @param right the index of the last element, inclusive, to be sorted
     */
    private static void doSort(double[] arr, int[] idxs, int left, int right) {
        // Use Quicksort on small arrays
        if (right - left < QUICKSORT_THRESHOLD) {
            sort(arr, idxs, left, right, true);
            return;
        }

        /*
         * Index run[i] is the start of i-th run
         * (ascending or descending sequence).
         */
        int[] run = new int[MAX_RUN_COUNT + 1];
        int count = 0; run[0] = left;

        // Check if the array is nearly sorted
        for (int k = left; k < right; run[count] = k) {
            if (arr[k] < arr[k + 1]) { // ascending
                while (++k <= right && arr[k - 1] <= arr[k]);
            } else if (arr[k] > arr[k + 1]) { // descending
                while (++k <= right && arr[k - 1] >= arr[k]);
                for (int lo = run[count] - 1, hi = k; ++lo < --hi; ) {
                    double t = arr[lo]; arr[lo] = arr[hi]; arr[hi] = t;
                    int v = idxs[lo]; idxs[lo] = idxs[hi]; idxs[hi] = v;
                }
            } else { // equal
                for (int m = MAX_RUN_LENGTH; ++k <= right && arr[k - 1] == arr[k]; ) {
                    if (--m == 0) {
                        sort(arr, idxs, left, right, true);
                        return;
                    }
                }
            }

            /*
             * The array is not highly structured,
             * use Quicksort instead of merge sort.
             */
            if (++count == MAX_RUN_COUNT) {
                sort(arr, idxs, left, right, true);
                return;
            }
        }

        // Check special cases
        if (run[count] == right++) { // The last run contains one element
            run[++count] = right;
        } else if (count == 1) { // The array is already sorted
            return;
        }

        /*
         * Create temporary array, which is used for merging.
         * Implementation note: variable "right" is increased by 1.
         */
        double[] tempArr; byte odd = 0; int[] tempIdxs;
        for (int n = 1; (n <<= 1) < count; odd ^= 1);

        if (odd == 0) {
            tempArr = arr; arr = new double[tempArr.length];
            tempIdxs = idxs; idxs = new int[tempIdxs.length];
            for (int i = left - 1; ++i < right; arr[i] = tempArr[i], idxs[i] = tempIdxs[i]);
        } else {
            tempArr = new double[arr.length];
            tempIdxs = new int[idxs.length];
        }

        // Merging
        for (int last; count > 1; count = last) {
            for (int k = (last = 0) + 2; k <= count; k += 2) {
                int hi = run[k], mi = run[k - 1];
                for (int i = run[k - 2], p = i, q = mi; i < hi; ++i) {
                    if (q >= hi || p < mi && arr[p] <= arr[q]) {
                        tempIdxs[i] = idxs[p];
                        tempArr[i] = arr[p++];
                    } else {
                        tempIdxs[i] = idxs[q];
                        tempArr[i] = arr[q++];
                    }
                }
                run[++last] = hi;
            }
            if ((count & 1) != 0) {
                for (int i = right, lo = run[count - 1]; --i >= lo;
                     tempArr[i] = arr[i], tempIdxs[i] = idxs[i]
                        );
                run[++last] = right;
            }
            double[] t = arr; arr = tempArr; tempArr = t;
            int[] v = idxs; idxs = tempIdxs; tempIdxs = v;
        }
    }

    /**
     * Sorts the specified range of the array by Dual-Pivot Quicksort.
     *
     * @param arr the array to be sorted
     * @param idxs the corresponding indices array, updated to match arr's sorted order
     * @param left the index of the first element, inclusive, to be sorted
     * @param right the index of the last element, inclusive, to be sorted
     * @param leftmost indicates if this part is the leftmost in the range
     */
    private static void sort(double[] arr, int[] idxs, int left, int right, boolean leftmost) {
        int length = right - left + 1;

        // Use insertion sort on tiny arrays
        if (length < INSERTION_SORT_THRESHOLD) {
            if (leftmost) {
                /*
                 * Traditional (without sentinel) insertion sort,
                 * optimized for server VM, is used in case of
                 * the leftmost part.
                 */
                for (int i = left, j = i; i < right; j = ++i) {
                    double ai = arr[i + 1];
                    int bi = idxs[i + 1];
                    while (ai < arr[j]) {
                        arr[j + 1] = arr[j];
                        idxs[j + 1] = idxs[j];
                        if (j-- == left) {
                            break;
                        }
                    }
                    arr[j + 1] = ai;
                    idxs[j + 1] = bi;
                }
            } else {
                /*
                 * Skip the longest ascending sequence.
                 */
                do {
                    if (left >= right) {
                        return;
                    }
                } while (arr[++left] >= arr[left - 1]);

                /*
                 * Every element from adjoining part plays the role
                 * of sentinel, therefore this allows us to avoid the
                 * left range check on each iteration. Moreover, we use
                 * the more optimized algorithm, so called pair insertion
                 * sort, which is faster (in the context of Quicksort)
                 * than traditional implementation of insertion sort.
                 */
                for (int k = left; ++left <= right; k = ++left) {
                    double a1 = arr[k], a2 = arr[left];
                    int b1 = idxs[k], b2 = idxs[left];

                    if (a1 < a2) {
                        a2 = a1; a1 = arr[left];
                        b2 = b1; b1 = idxs[left];
                    }
                    while (a1 < arr[--k]) {
                        arr[k + 2] = arr[k];
                        idxs[k + 2] = idxs[k];
                    }
                    arr[++k + 1] = a1;
                    idxs[k + 1] = b1;

                    while (a2 < arr[--k]) {
                        arr[k + 1] = arr[k];
                        idxs[k + 1] = idxs[k];
                    }
                    arr[k + 1] = a2;
                    idxs[k + 1] = b2;
                }
                double last = arr[right];
                int bLast = idxs[right];

                while (last < arr[--right]) {
                    arr[right + 1] = arr[right];
                    idxs[right + 1] = idxs[right];
                }
                arr[right + 1] = last;
                idxs[right + 1] = bLast;
            }
            return;
        }

        // Inexpensive approximation of length / 7
        int seventh = (length >> 3) + (length >> 6) + 1;

        /*
         * Sort five evenly spaced elements around (and including) the
         * center element in the range. These elements will be used for
         * pivot selection as described below. The choice for spacing
         * these elements was empirically determined to work well on
         * arr wide variety of inputs.
         */
        int e3 = (left + right) >>> 1; // The midpoint
        int e2 = e3 - seventh;
        int e1 = e2 - seventh;
        int e4 = e3 + seventh;
        int e5 = e4 + seventh;

        // Sort these elements using insertion sort
        if (arr[e2] < arr[e1]) {
            double t = arr[e2];
            int v = idxs[e2];

            arr[e2] = arr[e1];
            arr[e1] = t;
            idxs[e2] = idxs[e1];
            idxs[e1] = v;
        }

        if (arr[e3] < arr[e2]) {
            double t = arr[e3];
            int v = idxs[e3];

            arr[e3] = arr[e2];
            arr[e2] = t;
            idxs[e3] = idxs[e2];
            idxs[e2] = v;

            if (t < arr[e1]) {
                arr[e2] = arr[e1];
                arr[e1] = t;
                idxs[e2] = idxs[e1];
                idxs[e1] = v;
            }
        }

        if (arr[e4] < arr[e3]) {
            double t = arr[e4];
            int v = idxs[e4];

            arr[e4] = arr[e3];
            arr[e3] = t;
            idxs[e4] = idxs[e3];
            idxs[e3] = v;

            if (t < arr[e2]) {
                arr[e3] = arr[e2];
                arr[e2] = t;
                idxs[e3] = idxs[e2];
                idxs[e2] = v;

                if (t < arr[e1]) {
                    arr[e2] = arr[e1];
                    arr[e1] = t;
                    idxs[e2] = idxs[e1];
                    idxs[e1] = v;
                }
            }
        }

        if (arr[e5] < arr[e4]) {
            double t = arr[e5];
            int v = idxs[e5];

            arr[e5] = arr[e4];
            arr[e4] = t;
            idxs[e5] = idxs[e4];
            idxs[e4] = v;

            if (t < arr[e3]) {
                arr[e4] = arr[e3];
                arr[e3] = t;
                idxs[e4] = idxs[e3];
                idxs[e3] = v;

                if (t < arr[e2]) {
                    arr[e3] = arr[e2];
                    arr[e2] = t;
                    idxs[e3] = idxs[e2];
                    idxs[e2] = v;

                    if (t < arr[e1]) {
                        arr[e2] = arr[e1];
                        arr[e1] = t;
                        idxs[e2] = idxs[e1];
                        idxs[e1] = v;
                    }
                }
            }
        }

        // Pointers
        int less  = left;  // The index of the first element of center part
        int great = right; // The index before the first element of right part

        if (arr[e1] != arr[e2] && arr[e2] != arr[e3] && arr[e3] != arr[e4] && arr[e4] != arr[e5]) {
            /*
             * Use the second and fourth of the five sorted elements as pivots.
             * These values are inexpensive approximations of the first and
             * second terciles of the array. Note that pivot1 <= pivot2.
             */
            double pivot1 = arr[e2];
            double pivot2 = arr[e4];
            int idxPivot1 = idxs[e2];
            int idxPivot2 = idxs[e4];

            /*
             * The first and the last elements to be sorted are moved to the
             * locations formerly occupied by the pivots. When partitioning
             * is complete, the pivots are swapped back into their final
             * positions, and excluded from subsequent sorting.
             */
            arr[e2] = arr[left];
            arr[e4] = arr[right];
            idxs[e2] = idxs[left];
            idxs[e4] = idxs[right];

            /*
             * Skip elements, which are less or greater than pivot values.
             */
            while (arr[++less] < pivot1);
            while (arr[--great] > pivot2);

            /*
             * Partitioning:
             *
             *   left part           center part                   right part
             * +--------------------------------------------------------------+
             * |  < pivot1  |  pivot1 <= && <= pivot2  |    ?    |  > pivot2  |
             * +--------------------------------------------------------------+
             *               ^                          ^       ^
             *               |                          |       |
             *              less                        k     great
             *
             * Invariants:
             *
             *              all in (left, less)   < pivot1
             *    pivot1 <= all in [less, k)     <= pivot2
             *              all in (great, right) > pivot2
             *
             * Pointer k is the first index of ?-part.
             */
            outer:
            for (int k = less - 1; ++k <= great; ) {
                double ak = arr[k];
                int bk = idxs[k];
                if (ak < pivot1) { // Move arr[k] to left part
                    arr[k] = arr[less];
                    idxs[k] = idxs[less];
                    /*
                     * Here and below we use "arr[i] = b; i++;" instead
                     * of "arr[i++] = b;" due to performance issue.
                     */
                    arr[less] = ak;
                    idxs[less] = bk;
                    ++less;
                } else if (ak > pivot2) { // Move arr[k] to right part
                    while (arr[great] > pivot2) {
                        if (great-- == k) {
                            break outer;
                        }
                    }
                    if (arr[great] < pivot1) { // arr[great] <= pivot2
                        arr[k] = arr[less];
                        arr[less] = arr[great];
                        idxs[k] = idxs[less];
                        idxs[less] = idxs[great];
                        ++less;
                    } else { // pivot1 <= arr[great] <= pivot2
                        arr[k] = arr[great];
                        idxs[k] = idxs[great];
                    }
                    /*
                     * Here and below we use "arr[i] = b; i--;" instead
                     * of "arr[i--] = b;" due to performance issue.
                     */
                    arr[great] = ak;
                    idxs[great] = bk;
                    --great;
                }
            }

            // Swap pivots into their final positions
            arr[left]  = arr[less  - 1]; arr[less  - 1] = pivot1;
            arr[right] = arr[great + 1]; arr[great + 1] = pivot2;
            idxs[left]  = idxs[less  - 1]; idxs[less  - 1] = idxPivot1;
            idxs[right] = idxs[great + 1]; idxs[great + 1] = idxPivot2;

            // Sort left and right parts recursively, excluding known pivots
            sort(arr, idxs, left, less - 2, leftmost);
            sort(arr, idxs, great + 2, right, false);

            /*
             * If center part is too large (comprises > 4/7 of the array),
             * swap internal pivot values to ends.
             */
            if (less < e1 && e5 < great) {
                /*
                 * Skip elements, which are equal to pivot values.
                 */
                while (arr[less] == pivot1) {
                    ++less;
                }

                while (arr[great] == pivot2) {
                    --great;
                }

                /*
                 * Partitioning:
                 *
                 *   left part         center part                  right part
                 * +----------------------------------------------------------+
                 * | == pivot1 |  pivot1 < && < pivot2  |    ?    | == pivot2 |
                 * +----------------------------------------------------------+
                 *              ^                        ^       ^
                 *              |                        |       |
                 *             less                      k     great
                 *
                 * Invariants:
                 *
                 *              all in (*,  less) == pivot1
                 *     pivot1 < all in [less,  k)  < pivot2
                 *              all in (great, *) == pivot2
                 *
                 * Pointer k is the first index of ?-part.
                 */
                outer:
                for (int k = less - 1; ++k <= great; ) {
                    double ak = arr[k];
                    int bk = idxs[k];
                    if (ak == pivot1) { // Move arr[k] to left part
                        arr[k] = arr[less];
                        arr[less] = ak;
                        idxs[k] = idxs[less];
                        idxs[less] = bk;
                        ++less;
                    } else if (ak == pivot2) { // Move arr[k] to right part
                        while (arr[great] == pivot2) {
                            if (great-- == k) {
                                break outer;
                            }
                        }
                        if (arr[great] == pivot1) { // arr[great] < pivot2
                            arr[k] = arr[less];
                            idxs[k] = idxs[less];
                            /*
                             * Even though arr[great] equals to pivot1, the
                             * assignment arr[less] = pivot1 may be incorrect,
                             * if arr[great] and pivot1 are floating-point zeros
                             * of different signs. Therefore in float and
                             * double sorting methods we have to use more
                             * accurate assignment arr[less] = arr[great].
                             */
                            arr[less] = arr[great];
                            idxs[less] = idxs[great];
                            ++less;
                        } else { // pivot1 < arr[great] < pivot2
                            arr[k] = arr[great];
                            idxs[k] = idxs[great];
                        }
                        arr[great] = ak;
                        idxs[great] = bk;
                        --great;
                    }
                }
            }

            // Sort center part recursively
            sort(arr, idxs, less, great, false);

        } else { // Partitioning with one pivot
            /*
             * Use the third of the five sorted elements as pivot.
             * This value is inexpensive approximation of the median.
             */
            double pivot = arr[e3];

            /*
             * Partitioning degenerates to the traditional 3-way
             * (or "Dutch National Flag") schema:
             *
             *   left part    center part              right part
             * +-------------------------------------------------+
             * |  < pivot  |   == pivot   |     ?    |  > pivot  |
             * +-------------------------------------------------+
             *              ^              ^        ^
             *              |              |        |
             *             less            k      great
             *
             * Invariants:
             *
             *   all in (left, less)   < pivot
             *   all in [less, k)     == pivot
             *   all in (great, right) > pivot
             *
             * Pointer k is the first index of ?-part.
             */
            for (int k = less; k <= great; ++k) {
                if (arr[k] == pivot) {
                    continue;
                }
                double ak = arr[k];
                int bk = idxs[k];
                if (ak < pivot) { // Move arr[k] to left part
                    arr[k] = arr[less];
                    arr[less] = ak;
                    idxs[k] = idxs[less];
                    idxs[less] = bk;
                    ++less;
                } else { // arr[k] > pivot - Move arr[k] to right part
                    while (arr[great] > pivot) {
                        --great;
                    }
                    if (arr[great] < pivot) { // arr[great] <= pivot
                        arr[k] = arr[less];
                        arr[less] = arr[great];
                        idxs[k] = idxs[less];
                        idxs[less] = idxs[great];
                        ++less;
                    } else { // arr[great] == pivot
                        /*
                         * Even though arr[great] equals to pivot, the
                         * assignment arr[k] = pivot may be incorrect,
                         * if arr[great] and pivot are floating-point
                         * zeros of different signs. Therefore in float
                         * and double sorting methods we have to use
                         * more accurate assignment arr[k] = arr[great].
                         */
                        arr[k] = arr[great];
                        idxs[k] = idxs[great];
                    }
                    arr[great] = ak;
                    idxs[great] = bk;
                    --great;
                }
            }

            /*
             * Sort left and right parts recursively.
             * All elements from center part are equal
             * and, therefore, already sorted.
             */
            sort(arr, idxs, left, less - 1, leftmost);
            sort(arr, idxs, great + 1, right, false);
        }
    }
}
