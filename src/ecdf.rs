//! Empirical cumulative distribution function.

pub struct Ecdf<'s, T: Ord> {
    /// Checked at construction time to ensure it is never empty
    samples: &'s mut [T],
}

impl<'s, T: Ord + Clone> Ecdf<'s, T> {
    /// Construct a new representation of a cumulative distribution function for
    /// a given sample.
    ///
    /// The construction will involve computing a sort of the given sample
    /// and may be inefficient or completely prohibitive for large samples. This
    /// computation is amortized significantly if there is heavy use of the value
    /// function.
    ///
    /// Returns None if the sample is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kolmogorov_smirnov as ks;
    ///
    /// let mut samples = vec!(9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    /// let ecdf = ks::Ecdf::new(&mut samples).unwrap();
    /// ```
    pub fn new(samples: &mut [T]) -> Option<Ecdf<T>> {
        if samples.is_empty() {
            return None;
        }
        samples.sort_unstable();

        Some(Ecdf { samples })
    }

    /// Calculate a value of the empirical cumulative distribution function for
    /// a given sample.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kolmogorov_smirnov as ks;
    ///
    /// let mut samples = vec!(9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    /// let ecdf = ks::Ecdf::new(&mut samples).unwrap();
    /// assert_eq!(ecdf.value(4), 0.5);
    /// ```
    pub fn value(&self, t: T) -> f64 {
        let num_samples_leq_t = match self.samples.binary_search(&t) {
            Ok(mut index) => {
                // At least one sample is a t and we have the index of it. Need
                // to walk down the sorted samples until at last that == t.
                while index + 1 < self.samples.len() && self.samples[index + 1] == t {
                    index += 1;
                }

                // Compensate for 0-based indexing.
                index + 1
            }
            Err(index) => {
                // No sample is a t but if we had to put one in it would go at
                // index. This means all indices to the left have samples < t
                // and should be counted in the cdf proportion. We must take one
                // from index to get the last included sample but then we just
                // have to add one again to account for 0-based indexing.
                index
            }
        };

        num_samples_leq_t as f64 / self.samples.len() as f64
    }

    /// Calculate a percentile for the sample using the Nearest Rank method.
    ///
    /// Returns None if the percentile was less greater than 100.
    ///
    /// # Panics
    ///
    /// The percentile requested must be between 1 and 100 inclusive. In
    /// particular, there is no 0-percentile.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kolmogorov_smirnov as ks;
    ///
    /// let mut samples = vec!(9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    /// let ecdf = ks::Ecdf::new(&mut samples).unwrap();
    /// assert_eq!(ecdf.percentile(50).unwrap(), 4);
    /// ```
    pub fn percentile(&self, p: u8) -> Option<T> {
        if p == 0 || p > 100 {
            return None;
        }

        let rank = (p as f64 * self.samples.len() as f64 / 100.0).ceil() as usize;
        Some(self.samples[rank - 1].clone())
    }

    /// Calculate a permille for the sample using the Nearest Rank method.
    ///
    /// The permille requested must be between 1 and 1000 inclusive. In
    /// particular, there is no 0-permille. Returns None if an invalid
    /// permille is requested.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kolmogorov_smirnov as ks;
    ///
    /// let mut samples = vec!(9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    /// let ecdf = ks::Ecdf::new(&mut samples).unwrap();
    /// assert_eq!(ecdf.permille(500), Some(4));
    /// ```
    pub fn permille(&self, p: u16) -> Option<T> {
        if p == 0 || p > 1000 {
            return None;
        }

        let rank = (p as f64 * self.samples.len() as f64 / 1000.0).ceil() as usize;
        Some(self.samples[rank - 1].clone())
    }

    /// Calculate a rank element for the sample.
    ///
    /// The rank requested must be between 1 and the sample length inclusive. In
    /// particular, there is no 0-rank. Returns None if the rank provided is not present.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kolmogorov_smirnov as ks;
    ///
    /// let mut samples = vec!(9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    /// let ecdf = ks::Ecdf::new(&mut samples).unwrap();
    /// assert_eq!(ecdf.rank(5), Some(4));
    /// ```
    pub fn rank(&self, rank: usize) -> Option<T> {
        let length = self.samples.len();
        if 0 < rank && rank <= length {
            Some(self.samples[rank - 1].clone())
        } else {
            None
        }
    }

    /// Return the minimal element of the samples.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kolmogorov_smirnov as ks;
    ///
    /// let mut samples = vec!(9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    /// let ecdf = ks::Ecdf::new(&mut samples).unwrap();
    /// assert_eq!(ecdf.min(), 0);
    /// ```
    pub fn min(&self) -> T {
        self.samples[0].clone()
    }

    /// Return the maximal element of the samples.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kolmogorov_smirnov as ks;
    ///
    /// let mut samples = vec!(9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    /// let ecdf = ks::Ecdf::new(&mut samples).unwrap();
    /// assert_eq!(ecdf.max(), 9);
    /// ```
    pub fn max(&self) -> T {
        self.samples[self.samples.len() - 1].clone()
    }
}

/// Calculate a one-time value of the empirical cumulative distribution function
/// for a given sample.
///
/// Computational running time of this function is O(n) but does not amortize
/// across multiple calls like Ecdf<T>::value. This function should only be
/// used in the case that a small number of ECDF values are required for the
/// sample. Otherwise, Ecdf::new should be used to create a structure that
/// takes the upfront O(n log n) sort cost but calculates values in O(log n).
///
/// The sample set must be non-empty.
/// Returns None if the sample set is empty.
///
/// # Examples
///
/// ```
/// extern crate kolmogorov_smirnov as ks;
///
/// let samples = vec!(9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
/// let value = ks::ecdf(&samples, 4);
/// assert_eq!(value, Some(0.5));
/// ```
pub fn ecdf<T: Ord>(samples: &[T], t: T) -> Option<f64> {
    let mut num_samples_leq_t = 0;
    let mut length = 0;

    for sample in samples.iter() {
        length += 1;
        if *sample <= t {
            num_samples_leq_t += 1;
        }
    }

    if length == 0 {
        return None;
    }

    Some(num_samples_leq_t as f64 / length as f64)
}

/// Calculate a one-time percentile for a given sample using the Nearest Rank
/// method and Quick Select.
///
/// Computational running time of this function is O(n) but does not amortize
/// across multiple calls like Ecdf<T>::percentile. This function should only be
/// used in the case that a small number of percentiles are required for the
/// sample. Otherwise, Ecdf::new should be used to create a structure that
/// takes the upfront O(n log n) sort cost but calculates percentiles in O(1).
///
/// # Panics
///
/// The sample set must be non-empty.
///
/// The percentile requested must be between 1 and 100 inclusive. In particular,
/// there is no 0-percentile.
///
/// # Examples
///
/// ```
/// extern crate kolmogorov_smirnov as ks;
///
/// let samples = vec!(9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
/// let percentile = ks::percentile(&samples, 50);
/// assert_eq!(percentile, Some(4));
/// ```
pub fn percentile<T: Ord + Clone>(samples: &[T], p: u8) -> Option<T> {
    if 0 == p || p > 100 {
        return None;
    }

    let length = samples.len();
    if length == 0 {
        return None;
    }

    let r = (p as f64 * length as f64 / 100.0).ceil() as usize;

    rank(samples, r)
}

/// Calculate a one-time permille for a given sample using the Nearest Rank
/// method and Quick Select.
///
/// Computational running time of this function is O(n) but does not amortize
/// across multiple calls like Ecdf<T>::permille. This function should only be
/// used in the case that a small number of permilles are required for the
/// sample. Otherwise, Ecdf::new should be used to create a structure that
/// takes the upfront O(n log n) sort cost but calculates permilles in O(1).
///
/// # Panics
///
/// The sample set must be non-empty.
///
/// The permille requested must be between 1 and 1000 inclusive. In particular,
/// there is no 0-permille.
///
/// # Examples
///
/// ```
/// extern crate kolmogorov_smirnov as ks;
///
/// let samples = vec!(9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
/// let permille = ks::permille(&samples, 500);
/// assert_eq!(permille, Some(4));
/// ```
pub fn permille<T: Ord + Clone>(samples: &[T], p: u16) -> Option<T> {
    if p == 0 || p > 1000 {
        return None;
    }

    let length = samples.len();
    if length == 0 {
        return None;
    }

    let r = (p as f64 * length as f64 / 1000.0).ceil() as usize;

    rank(samples, r)
}

/// Calculate a one-time rank for a given sample using Quick Select.
///
/// Computational running time of this function is O(n) and does not amortize
/// across multiple calls. This function should only be used in the case that a
/// small number of ranks are required for the sample.
///
///
/// # Invariants
///
/// The sample set must be non-empty.
///
/// The rank requested must be between 1 and the sample length inclusive. In
/// particular, there is no 0-rank.
///
/// Returns None if any invariant is not met.
///
/// # Examples
///
/// ```
/// extern crate kolmogorov_smirnov as ks;
///
/// let samples = vec!(9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
/// let rank = ks::rank(&samples, 5);
/// assert_eq!(rank, Some(4));
/// ```
pub fn rank<T: Ord + Clone>(samples: &[T], rank: usize) -> Option<T> {
    let length = samples.len();
    if length == 0 || rank == 0 || rank > length {
        return None;
    }

    // Quick Select the element at rank.

    let mut samples: Vec<T> = samples.to_vec();
    let mut low = 0;
    let mut high = length;

    loop {
        // Semantic check
        if low >= high {
            return None;
        }

        let pivot = samples[low].clone();

        if low >= high - 1 {
            return Some(pivot);
        }

        // First determine if the rank item is less than the pivot.

        // Organise samples so that all items less than pivot are to the left,
        // `bottom` is the number of items less than pivot.

        let mut bottom = low;
        let mut top = high - 1;

        while bottom < top {
            while bottom < top && samples[bottom] < pivot {
                bottom += 1;
            }
            while bottom < top && samples[top] >= pivot {
                top -= 1;
            }

            if bottom < top {
                samples.swap(bottom, top);
            }
        }

        if rank <= bottom {
            // Rank item is less than pivot. Exclude pivot and larger items.
            high = bottom;
        } else {
            // Rank item is pivot or in the larger set. Exclude smaller items.
            low = bottom;

            // Next, determine if the pivot is the rank item.

            // Organise samples so that all items less than or equal to pivot
            // are to the left, `bottom` is the number of items less than or
            // equal to pivot. Since the left is already less than the pivot,
            // this just requires moving the pivots left also.

            let mut bottom = low;
            let mut top = high - 1;

            while bottom < top {
                while bottom < top && samples[bottom] == pivot {
                    bottom += 1;
                }
                while bottom < top && samples[top] != pivot {
                    top -= 1;
                }

                if bottom < top {
                    samples.swap(bottom, top);
                }
            }

            // Is pivot the rank item?

            if rank <= bottom {
                return Some(pivot);
            }

            // Rank item is greater than pivot. Exclude pivot and smaller items.
            low = bottom;
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate quickcheck;
    extern crate rand;

    use self::quickcheck::{Arbitrary, Gen, QuickCheck, StdGen, TestResult, Testable};
    use super::{ecdf, percentile, permille, rank, Ecdf};
    use std::cmp;
    use std::usize;

    fn check<A: Testable>(f: A) {
        let g = StdGen::new(rand::thread_rng(), usize::MAX);
        QuickCheck::new().gen(g).quickcheck(f);
    }

    /// Wrapper for generating sample data with QuickCheck.
    ///
    /// Samples must be non-empty sequences of u64 values.
    #[derive(Debug, Clone)]
    struct Samples {
        vec: Vec<u64>,
    }

    impl Arbitrary for Samples {
        fn arbitrary<G: Gen>(g: &mut G) -> Samples {
            // Limit size of generated sample set to 1024
            let max = cmp::min(g.size(), 1024);

            let size = g.gen_range(1, max);
            let vec = (0..size).map(|_| u64::arbitrary(g)).collect();

            Samples { vec: vec }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Samples>> {
            let vec: Vec<u64> = self.vec.clone();
            let shrunk: Box<dyn Iterator<Item = Vec<u64>>> = vec.shrink();

            Box::new(shrunk.filter(|v| v.len() > 0).map(|v| Samples { vec: v }))
        }
    }

    /// Wrapper for generating percentile query value data with QuickCheck.
    ///
    /// Percentile must be u8 between 1 and 100 inclusive.
    #[derive(Debug, Clone)]
    struct Percentile {
        val: u8,
    }

    impl Arbitrary for Percentile {
        fn arbitrary<G: Gen>(g: &mut G) -> Percentile {
            let val = g.gen_range(1, 101) as u8;

            Percentile { val: val }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Percentile>> {
            let shrunk: Box<dyn Iterator<Item = u8>> = self.val.shrink();

            Box::new(
                shrunk
                    .filter(|&v| 0u8 < v && v <= 100u8)
                    .map(|v| Percentile { val: v }),
            )
        }
    }

    /// Wrapper for generating permille query value data with QuickCheck.
    ///
    /// Percentile must be u16 between 1 and 1000 inclusive.
    #[derive(Debug, Clone)]
    struct Permille {
        val: u16,
    }

    impl Arbitrary for Permille {
        fn arbitrary<G: Gen>(g: &mut G) -> Permille {
            let val = g.gen_range(1, 1001) as u16;

            Permille { val: val }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Permille>> {
            let shrunk: Box<dyn Iterator<Item = u16>> = self.val.shrink();

            Box::new(
                shrunk
                    .filter(|&v| 0u16 < v && v <= 1000u16)
                    .map(|v| Permille { val: v }),
            )
        }
    }

    #[test]
    fn single_use_ecdf_none_on_empty_samples_set() {
        let xs: Vec<u64> = vec![];
        assert!(ecdf(&xs, 0).is_none());
    }

    #[test]
    fn multiple_use_ecdf_none_on_empty_samples_set() {
        let mut xs: Vec<u64> = vec![];
        assert!(Ecdf::new(&mut xs).is_none());
    }

    #[test]
    fn single_use_ecdf_between_zero_and_one() {
        fn prop(xs: Samples, val: u64) -> bool {
            let actual = ecdf(&xs.vec, val).unwrap();

            0.0 <= actual && actual <= 1.0
        }

        check(prop as fn(Samples, u64) -> bool);
    }

    #[test]
    fn multiple_use_ecdf_between_zero_and_one() {
        fn prop(mut xs: Samples, val: u64) -> bool {
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.value(val);

            0.0 <= actual && actual <= 1.0
        }

        check(prop as fn(Samples, u64) -> bool);
    }

    #[test]
    fn single_use_ecdf_is_an_increasing_function() {
        fn prop(xs: Samples, val: u64) -> bool {
            let actual = ecdf(&xs.vec, val);

            ecdf(&xs.vec, val - 1) <= actual && actual <= ecdf(&xs.vec, val + 1)
        }

        check(prop as fn(Samples, u64) -> bool);
    }

    #[test]
    fn multiple_use_ecdf_is_an_increasing_function() {
        fn prop(mut xs: Samples, val: u64) -> bool {
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.value(val);

            ecdf.value(val - 1) <= actual && actual <= ecdf.value(val + 1)
        }

        check(prop as fn(Samples, u64) -> bool);
    }

    #[test]
    fn single_use_ecdf_sample_min_minus_one_is_zero() {
        fn prop(xs: Samples) -> bool {
            let &min = xs.vec.iter().min().unwrap();

            ecdf(&xs.vec, min - 1).unwrap() == 0.0
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn multiple_use_ecdf_sample_min_minus_one_is_zero() {
        fn prop(mut xs: Samples) -> bool {
            let &min = xs.vec.iter().min().unwrap();
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            ecdf.value(min - 1) == 0.0
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn single_use_ecdf_sample_max_is_one() {
        fn prop(xs: Samples) -> bool {
            let &max = xs.vec.iter().max().unwrap();

            ecdf(&xs.vec, max).unwrap() == 1.0
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn multiple_use_ecdf_sample_max_is_one() {
        fn prop(mut xs: Samples) -> bool {
            let &max = xs.vec.iter().max().unwrap();
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            ecdf.value(max) == 1.0
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn single_use_ecdf_sample_val_is_num_samples_leq_val_div_length() {
        fn prop(xs: Samples) -> bool {
            let &val = xs.vec.first().unwrap();
            let num_samples = xs.vec.iter().filter(|&&x| x <= val).count();
            let expected = num_samples as f64 / xs.vec.len() as f64;

            ecdf(&xs.vec, val).unwrap() == expected
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn multiple_use_ecdf_sample_val_is_num_samples_leq_val_div_length() {
        fn prop(mut xs: Samples) -> bool {
            let &val = xs.vec.first().unwrap();
            let num_samples = xs.vec.iter().filter(|&&x| x <= val).count();
            let expected = num_samples as f64 / xs.vec.len() as f64;

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            ecdf.value(val) == expected
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn single_use_ecdf_non_sample_val_is_num_samples_leq_val_div_length() {
        fn prop(xs: Samples, val: u64) -> TestResult {
            let length = xs.vec.len();

            if xs.vec.iter().any(|&x| x == val) {
                // Discard Vec containing val.
                return TestResult::discard();
            }

            let num_samples = xs.vec.iter().filter(|&&x| x <= val).count();
            let expected = num_samples as f64 / length as f64;

            let actual = ecdf(&xs.vec, val).unwrap();

            TestResult::from_bool(actual == expected)
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn multiple_use_ecdf_non_sample_val_is_num_samples_leq_val_div_length() {
        fn prop(mut xs: Samples, val: u64) -> TestResult {
            let length = xs.vec.len();

            if xs.vec.iter().any(|&x| x == val) {
                // Discard Vec containing val.
                return TestResult::discard();
            }

            let num_samples = xs.vec.iter().filter(|&&x| x <= val).count();
            let expected = num_samples as f64 / length as f64;

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            TestResult::from_bool(ecdf.value(val) == expected)
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn single_and_multiple_use_ecdf_agree() {
        fn prop(mut xs: Samples, val: u64) -> bool {
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();

            multiple_use.value(val) == ecdf(&xs.vec, val).unwrap()
        }

        check(prop as fn(Samples, u64) -> bool);
    }

    #[test]
    fn single_use_percentiles_none_on_zero_percentile() {
        let xs: Vec<u64> = vec![0];

        assert!(percentile(&xs, 0).is_none());
    }

    #[test]
    fn single_use_percentiles_none_on_101_percentile() {
        let xs: Vec<u64> = vec![0];

        assert!(percentile(&xs, 101).is_none());
    }

    #[test]
    fn multiple_use_percentiles_none_on_zero_percentile() {
        let mut xs: Vec<u64> = vec![0];
        let ecdf = Ecdf::new(&mut xs).unwrap();

        assert!(ecdf.percentile(0).is_none());
    }

    #[test]
    fn multiple_use_percentiles_none_on_101_percentile() {
        let mut xs: Vec<u64> = vec![0];
        let ecdf = Ecdf::new(&mut xs).unwrap();

        assert!(ecdf.percentile(101).is_none());
    }

    #[test]
    fn single_use_percentile_between_samples_min_and_max() {
        fn prop(xs: Samples, p: Percentile) -> bool {
            let &min = xs.vec.iter().min().unwrap();
            let &max = xs.vec.iter().max().unwrap();

            let actual = percentile(&xs.vec, p.val).unwrap();

            min <= actual && actual <= max
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn single_use_percentile_is_an_increasing_function() {
        fn prop(xs: Samples, p: Percentile) -> bool {
            let smaller = cmp::max(p.val - 1, 1);
            let larger = cmp::min(p.val + 1, 100);

            let actual = percentile(&xs.vec, p.val);

            percentile(&xs.vec, smaller) <= actual && actual <= percentile(&xs.vec, larger)
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn single_use_percentile_100_is_sample_max() {
        fn prop(xs: Samples) -> bool {
            let &max = xs.vec.iter().max().unwrap();

            percentile(&xs.vec, 100).unwrap() == max
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn multiple_use_percentile_between_samples_min_and_max() {
        fn prop(mut xs: Samples, p: Percentile) -> bool {
            let &min = xs.vec.iter().min().unwrap();
            let &max = xs.vec.iter().max().unwrap();

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.percentile(p.val).unwrap();

            min <= actual && actual <= max
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn multiple_use_percentile_is_an_increasing_function() {
        fn prop(mut xs: Samples, p: Percentile) -> bool {
            let smaller = cmp::max(p.val - 1, 1);
            let larger = cmp::min(p.val + 1, 100);

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.percentile(p.val).unwrap();

            ecdf.percentile(smaller).unwrap() <= actual
                && actual <= ecdf.percentile(larger).unwrap()
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn multiple_use_percentile_100_is_sample_max() {
        fn prop(mut xs: Samples) -> bool {
            let &max = xs.vec.iter().max().unwrap();
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            ecdf.percentile(100).unwrap() == max
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn single_use_ecdf_followed_by_single_use_percentile_is_leq_original_value() {
        fn prop(xs: Samples, val: u64) -> TestResult {
            let actual = ecdf(&xs.vec, val).unwrap();

            let p = (actual * 100.0).floor() as u8;

            match p {
                0 => {
                    // val is below the first percentile threshold. Can't
                    // calculate 0-percentile value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all percentiles of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and percentile(100) == 0.
                    let single_use = percentile(&xs.vec, p).unwrap();
                    TestResult::from_bool(single_use <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn single_use_ecdf_followed_by_multiple_use_percentile_is_leq_original_value() {
        fn prop(mut xs: Samples, val: u64) -> TestResult {
            let actual = ecdf(&xs.vec, val).unwrap();

            let p = (actual * 100.0).floor() as u8;

            match p {
                0 => {
                    // val is below the first percentile threshold. Can't
                    // calculate 0-percentile value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all percentiles of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and percentile(100) == 0.
                    let multiple_use = Ecdf::new(&mut xs.vec).unwrap();
                    TestResult::from_bool(multiple_use.percentile(p).unwrap() <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn multiple_use_ecdf_followed_by_single_use_percentile_is_leq_original_value() {
        fn prop(mut xs: Samples, val: u64) -> TestResult {
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.value(val);

            let p = (actual * 100.0).floor() as u8;

            match p {
                0 => {
                    // val is below the first percentile threshold. Can't
                    // calculate 0-percentile value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all percentiles of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and percentile(100) == 0.
                    let single_use = percentile(&xs.vec, p).unwrap();
                    TestResult::from_bool(single_use <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn multiple_use_ecdf_followed_by_multiple_use_percentile_is_leq_original_value() {
        fn prop(mut xs: Samples, val: u64) -> TestResult {
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.value(val);

            let p = (actual * 100.0).floor() as u8;

            match p {
                0 => {
                    // val is below the first percentile threshold. Can't
                    // calculate 0-percentile value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all percentiles of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and percentile(100) == 0.
                    TestResult::from_bool(ecdf.percentile(p).unwrap() <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn single_use_percentile_followed_by_single_use_ecdf_is_geq_original_value() {
        fn prop(xs: Samples, p: Percentile) -> bool {
            let actual = percentile(&xs.vec, p.val).unwrap();

            // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
            // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
            p.val as f64 / 100.0 <= ecdf(&xs.vec, actual).unwrap()
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn single_use_percentile_followed_by_multiple_use_ecdf_is_geq_original_value() {
        fn prop(mut xs: Samples, p: Percentile) -> bool {
            let actual = percentile(&xs.vec, p.val).unwrap();

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
            // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
            p.val as f64 / 100.0 <= ecdf.value(actual)
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn multiple_use_percentile_followed_by_single_use_ecdf_is_geq_original_value() {
        fn prop(mut xs: Samples, p: Percentile) -> bool {
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();
            let actual = multiple_use.percentile(p.val).unwrap();

            // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
            // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
            p.val as f64 / 100.0 <= ecdf(&xs.vec, actual).unwrap()
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn multiple_use_percentile_followed_by_multiple_use_ecdf_is_geq_original_value() {
        fn prop(mut xs: Samples, p: Percentile) -> bool {
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.percentile(p.val).unwrap();

            // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
            // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
            p.val as f64 / 100.0 <= ecdf.value(actual)
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn single_and_multiple_use_percentile_agree() {
        fn prop(mut xs: Samples, p: Percentile) -> bool {
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();

            multiple_use.percentile(p.val).unwrap() == percentile(&xs.vec, p.val).unwrap()
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn single_use_permilles_none_on_zero_permille() {
        let xs: Vec<u64> = vec![0];

        assert!(permille(&xs, 0).is_none());
    }

    #[test]
    fn single_use_permilles_none_on_1001_permille() {
        let xs: Vec<u64> = vec![0];

        assert!(permille(&xs, 1001).is_none());
    }

    #[test]
    fn multiple_use_permilles_none_on_zero_permille() {
        let mut xs: Vec<u64> = vec![0];
        let ecdf = Ecdf::new(&mut xs).unwrap();

        assert!(ecdf.permille(0).is_none());
    }

    #[test]
    fn multiple_use_permilles_none_on_1001_permille() {
        let mut xs: Vec<u64> = vec![0];
        let ecdf = Ecdf::new(&mut xs).unwrap();

        assert!(ecdf.permille(1001).is_none());
    }

    #[test]
    fn single_use_permille_between_samples_min_and_max() {
        fn prop(xs: Samples, p: Permille) -> bool {
            let &min = xs.vec.iter().min().unwrap();
            let &max = xs.vec.iter().max().unwrap();

            let actual = permille(&xs.vec, p.val).unwrap();

            min <= actual && actual <= max
        }

        check(prop as fn(Samples, Permille) -> bool);
    }

    #[test]
    fn single_use_permille_is_an_increasing_function() {
        fn prop(xs: Samples, p: Permille) -> bool {
            let smaller = cmp::max(p.val - 1, 1);
            let larger = cmp::min(p.val + 1, 1000);

            let actual = permille(&xs.vec, p.val);

            permille(&xs.vec, smaller) <= actual && actual <= permille(&xs.vec, larger)
        }

        check(prop as fn(Samples, Permille) -> bool);
    }

    #[test]
    fn single_use_permille_1000_is_sample_max() {
        fn prop(xs: Samples) -> bool {
            let &max = xs.vec.iter().max().unwrap();

            permille(&xs.vec, 1000).unwrap() == max
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn multiple_use_permille_between_samples_min_and_max() {
        fn prop(mut xs: Samples, p: Permille) -> bool {
            let &min = xs.vec.iter().min().unwrap();
            let &max = xs.vec.iter().max().unwrap();

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.permille(p.val).unwrap();

            min <= actual && actual <= max
        }

        check(prop as fn(Samples, Permille) -> bool);
    }

    #[test]
    fn multiple_use_permille_is_an_increasing_function() {
        fn prop(mut xs: Samples, p: Permille) -> bool {
            let smaller = cmp::max(p.val - 1, 1);
            let larger = cmp::min(p.val + 1, 1000);

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.permille(p.val).unwrap();

            ecdf.permille(smaller).unwrap() <= actual && actual <= ecdf.permille(larger).unwrap()
        }

        check(prop as fn(Samples, Permille) -> bool);
    }

    #[test]
    fn multiple_use_permille_1000_is_sample_max() {
        fn prop(mut xs: Samples) -> bool {
            let &max = xs.vec.iter().max().unwrap();
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            ecdf.permille(1000).unwrap() == max
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn single_use_ecdf_followed_by_single_use_permille_is_leq_original_value() {
        fn prop(xs: Samples, val: u64) -> TestResult {
            let actual = ecdf(&xs.vec, val).unwrap();

            let p = (actual * 1000.0).floor() as u16;

            match p {
                0 => {
                    // val is below the first permille threshold. Can't
                    // calculate 0-permille value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all permilles of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and permille(1000) == 0.
                    let single_use = permille(&xs.vec, p).unwrap();
                    TestResult::from_bool(single_use <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn single_use_ecdf_followed_by_multiple_use_permille_is_leq_original_value() {
        fn prop(mut xs: Samples, val: u64) -> TestResult {
            let actual = ecdf(&xs.vec, val).unwrap();

            let p = (actual * 1000.0).floor() as u16;

            match p {
                0 => {
                    // val is below the first permille threshold. Can't
                    // calculate 0-permille value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all permilles of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and permille(1000) == 0.
                    let multiple_use = Ecdf::new(&mut xs.vec).unwrap();
                    TestResult::from_bool(multiple_use.permille(p).unwrap() <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn multiple_use_ecdf_followed_by_single_use_permille_is_leq_original_value() {
        fn prop(mut xs: Samples, val: u64) -> TestResult {
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.value(val);

            let p = (actual * 1000.0).floor() as u16;

            match p {
                0 => {
                    // val is below the first permille threshold. Can't
                    // calculate 0-permille value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all permilles of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and permille(1000) == 0.
                    let single_use = permille(&xs.vec, p).unwrap();
                    TestResult::from_bool(single_use <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn multiple_use_ecdf_followed_by_multiple_use_permille_is_leq_original_value() {
        fn prop(mut xs: Samples, val: u64) -> TestResult {
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.value(val);

            let p = (actual * 1000.0).floor() as u16;

            match p {
                0 => {
                    // val is below the first permille threshold. Can't
                    // calculate 0-permille value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all permilles of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and permille(1000) == 0.
                    TestResult::from_bool(ecdf.permille(p).unwrap() <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn single_use_permille_followed_by_single_use_ecdf_is_geq_original_value() {
        fn prop(xs: Samples, p: Permille) -> bool {
            let actual = permille(&xs.vec, p.val).unwrap();

            // Not equal because e.g. 1- through 500-permilles of [0, 1] are 0.
            // So original value of 1 gives permille == 0 and ecdf(0) == 0.5.
            p.val as f64 / 1000.0 <= ecdf(&xs.vec, actual).unwrap()
        }

        check(prop as fn(Samples, Permille) -> bool);
    }

    #[test]
    fn single_use_permille_followed_by_multiple_use_ecdf_is_geq_original_value() {
        fn prop(mut xs: Samples, p: Permille) -> bool {
            let actual = permille(&xs.vec, p.val).unwrap();

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            // Not equal because e.g. 1- through 500-permilles of [0, 1] are 0.
            // So original value of 1 gives permille == 0 and ecdf(0) == 0.5.
            p.val as f64 / 1000.0 <= ecdf.value(actual)
        }

        check(prop as fn(Samples, Permille) -> bool);
    }

    #[test]
    fn multiple_use_permille_followed_by_single_use_ecdf_is_geq_original_value() {
        fn prop(mut xs: Samples, p: Permille) -> bool {
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();
            let actual = multiple_use.permille(p.val).unwrap();

            // Not equal because e.g. 1- through 500-permilles of [0, 1] are 0.
            // So original value of 1 gives permille == 0 and ecdf(0) == 0.5.
            p.val as f64 / 1000.0 <= ecdf(&xs.vec, actual).unwrap()
        }

        check(prop as fn(Samples, Permille) -> bool);
    }

    #[test]
    fn multiple_use_permille_followed_by_multiple_use_ecdf_is_geq_original_value() {
        fn prop(mut xs: Samples, p: Permille) -> bool {
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.permille(p.val).unwrap();

            // Not equal because e.g. 1- through 500-permilles of [0, 1] are 0.
            // So original value of 1 gives permille == 0 and ecdf(0) == 0.5.
            p.val as f64 / 1000.0 <= ecdf.value(actual)
        }

        check(prop as fn(Samples, Permille) -> bool);
    }

    #[test]
    fn single_and_multiple_use_permille_agree() {
        fn prop(mut xs: Samples, p: Permille) -> bool {
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();

            multiple_use.permille(p.val).unwrap() == permille(&xs.vec, p.val).unwrap()
        }

        check(prop as fn(Samples, Permille) -> bool);
    }

    #[test]
    fn single_use_percentile_and_single_use_permille_agree() {
        fn prop(mut xs: Samples, p: Percentile) -> bool {
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();

            multiple_use.percentile(p.val).unwrap() == permille(&xs.vec, p.val as u16 * 10).unwrap()
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn single_use_percentile_and_multiple_use_permille_agree() {
        fn prop(mut xs: Samples, p: Percentile) -> bool {
            let nrqs_percentile = percentile(&xs.vec, p.val).unwrap();
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();
            nrqs_percentile == multiple_use.permille(p.val as u16 * 10).unwrap()
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn multiple_use_percentile_and_single_use_permille_agree() {
        fn prop(mut xs: Samples, p: Percentile) -> bool {
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();

            multiple_use.percentile(p.val).unwrap() == permille(&xs.vec, p.val as u16 * 10).unwrap()
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn multiple_use_percentile_and_multiple_use_permille_agree() {
        fn prop(mut xs: Samples, p: Percentile) -> bool {
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();

            multiple_use.percentile(p.val).unwrap()
                == multiple_use.permille(p.val as u16 * 10).unwrap()
        }

        check(prop as fn(Samples, Percentile) -> bool);
    }

    #[test]
    fn single_use_rank_none_on_zero_rank() {
        let xs: Vec<u64> = vec![0];

        assert!(rank(&xs, 0).is_none());
    }

    #[test]
    fn single_use_rank_none_on_too_large_rank() {
        let xs: Vec<u64> = vec![0];

        assert!(rank(&xs, 2).is_none());
    }

    #[test]
    fn multiple_use_rank_none_on_zero_rank() {
        let mut xs: Vec<u64> = vec![0];
        let ecdf = Ecdf::new(&mut xs).unwrap();

        assert!(ecdf.rank(0).is_none());
    }

    #[test]
    fn multiple_use_rank_none_on_too_large_rank() {
        let mut xs: Vec<u64> = vec![0];
        let ecdf = Ecdf::new(&mut xs).unwrap();

        assert!(ecdf.rank(2).is_none());
    }

    #[test]
    fn single_use_rank_between_samples_min_and_max() {
        fn prop(xs: Samples, r: usize) -> bool {
            let length = xs.vec.len();
            let &min = xs.vec.iter().min().unwrap();
            let &max = xs.vec.iter().max().unwrap();

            let x = r % length + 1;
            let actual = rank(&xs.vec, x).unwrap();
            min <= actual && actual <= max
        }

        check(prop as fn(Samples, usize) -> bool);
    }

    #[test]
    fn single_use_rank_is_an_increasing_function() {
        fn prop(xs: Samples, r: usize) -> bool {
            let length = xs.vec.len();
            let x = r % length + 1;

            let smaller = cmp::max(x - 1, 1);
            let larger = cmp::min(x + 1, length);

            let actual = rank(&xs.vec, x);

            rank(&xs.vec, smaller) <= actual && actual <= rank(&xs.vec, larger)
        }

        check(prop as fn(Samples, usize) -> bool);
    }

    #[test]
    fn single_use_rank_1_is_sample_min() {
        fn prop(xs: Samples) -> bool {
            let &min = xs.vec.iter().min().unwrap();

            rank(&xs.vec, 1).unwrap() == min
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn single_use_rank_length_is_sample_max() {
        fn prop(xs: Samples) -> bool {
            let &max = xs.vec.iter().max().unwrap();

            rank(&xs.vec, xs.vec.len()).unwrap() == max
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn multiple_use_rank_between_samples_min_and_max() {
        fn prop(mut xs: Samples, r: usize) -> bool {
            let length = xs.vec.len();
            let &min = xs.vec.iter().min().unwrap();
            let &max = xs.vec.iter().max().unwrap();

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            let x = r % length + 1;
            let actual = ecdf.rank(x).unwrap();
            min <= actual && actual <= max
        }

        check(prop as fn(Samples, usize) -> bool);
    }

    #[test]
    fn multiple_use_rank_is_an_increasing_function() {
        fn prop(mut xs: Samples, r: usize) -> bool {
            let length = xs.vec.len();
            let x = r % length + 1;

            let smaller = cmp::max(x - 1, 1);
            let larger = cmp::min(x + 1, length);

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.rank(x).unwrap();

            ecdf.rank(smaller).unwrap() <= actual && actual <= ecdf.rank(larger).unwrap()
        }

        check(prop as fn(Samples, usize) -> bool);
    }

    #[test]
    fn multiple_use_rank_1_is_sample_min() {
        fn prop(mut xs: Samples) -> bool {
            let &min = xs.vec.iter().min().unwrap();
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            ecdf.rank(1).unwrap() == min
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn multiple_use_rank_length_is_sample_max() {
        fn prop(mut xs: Samples) -> bool {
            let &max = xs.vec.iter().max().unwrap();
            let xs_len = xs.vec.len();
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            ecdf.rank(xs_len).unwrap() == max
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn single_use_ecdf_followed_by_single_use_rank_is_leq_original_value() {
        fn prop(xs: Samples, val: u64) -> TestResult {
            let length = xs.vec.len();
            let actual = ecdf(&xs.vec, val).unwrap();

            let p = (actual * length as f64).floor() as usize;

            match p {
                0 => {
                    // val is below the first rank threshold. Can't
                    // calculate 0-rank value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all ranks of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and rank(1) == 0.
                    let single_use = rank(&xs.vec, p).unwrap();
                    TestResult::from_bool(single_use <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn single_use_ecdf_followed_by_multiple_use_rank_is_leq_original_value() {
        fn prop(mut xs: Samples, val: u64) -> TestResult {
            let length = xs.vec.len();
            let actual = ecdf(&xs.vec, val).unwrap();

            let p = (actual * length as f64).floor() as usize;

            match p {
                0 => {
                    // val is below the first rank threshold. Can't
                    // calculate 0-rank value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all ranks of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and rank(1) == 0.
                    let multiple_use = Ecdf::new(&mut xs.vec).unwrap();
                    TestResult::from_bool(multiple_use.rank(p).unwrap() <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn multiple_use_ecdf_followed_by_single_use_rank_is_leq_original_value() {
        fn prop(mut xs: Samples, val: u64) -> TestResult {
            let length = xs.vec.len();
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.value(val);

            let p = (actual * length as f64).floor() as usize;

            match p {
                0 => {
                    // val is below the first rank threshold. Can't
                    // calculate 0-rank value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all ranks of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and rank(1) == 0.
                    let single_use = rank(&xs.vec, p).unwrap();
                    TestResult::from_bool(single_use <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn multiple_use_ecdf_followed_by_multiple_use_rank_is_leq_original_value() {
        fn prop(mut xs: Samples, val: u64) -> TestResult {
            let length = xs.vec.len();
            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.value(val);

            let p = (actual * length as f64).floor() as usize;

            match p {
                0 => {
                    // val is below the first rank threshold. Can't
                    // calculate 0-rank value so discard.
                    TestResult::discard()
                }
                _ => {
                    // Not equal because e.g. all ranks of [0] are 0. So
                    // value of 1 gives ecdf == 1.0 and rank(1) == 0.
                    TestResult::from_bool(ecdf.rank(p).unwrap() <= val)
                }
            }
        }

        check(prop as fn(Samples, u64) -> TestResult);
    }

    #[test]
    fn single_use_rank_followed_by_single_use_ecdf_is_geq_original_value() {
        fn prop(xs: Samples, r: usize) -> bool {
            let length = xs.vec.len();
            let x = r % length + 1;

            let actual = rank(&xs.vec, x).unwrap();

            // Not equal because e.g. all ranks of [0, 0] are 0. So
            // rank(1) == 0 and value of 0 gives ecdf == 1.0.
            (x as f64 / length as f64) <= ecdf(&xs.vec, actual).unwrap()
        }

        assert!(prop(Samples { vec: vec![0, 0] }, 0));
        check(prop as fn(Samples, usize) -> bool);
    }

    #[test]
    fn single_use_rank_followed_by_multiple_use_ecdf_is_geq_original_value() {
        fn prop(mut xs: Samples, r: usize) -> bool {
            let length = xs.vec.len();
            let x = r % length + 1;

            let actual = rank(&xs.vec, x).unwrap();

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();

            // Not equal because e.g. all ranks of [0, 0] are 0. So
            // rank(1) == 0 and value of 0 gives ecdf == 1.0.
            (x as f64 / length as f64) <= ecdf.value(actual)
        }

        assert!(prop(Samples { vec: vec![0, 0] }, 0));
        check(prop as fn(Samples, usize) -> bool);
    }

    #[test]
    fn multiple_use_rank_followed_by_single_use_ecdf_is_geq_original_value() {
        fn prop(mut xs: Samples, r: usize) -> bool {
            let length = xs.vec.len();
            let x = r % length + 1;

            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();
            let actual = multiple_use.rank(x).unwrap();

            // Not equal because e.g. all ranks of [0, 0] are 0. So
            // rank(1) == 0 and value of 0 gives ecdf == 1.0.
            (x as f64 / length as f64) <= ecdf(&xs.vec, actual).unwrap()
        }

        assert!(prop(Samples { vec: vec![0, 0] }, 0));
        check(prop as fn(Samples, usize) -> bool);
    }

    #[test]
    fn multiple_use_rank_followed_by_multiple_use_ecdf_is_geq_original_value() {
        fn prop(mut xs: Samples, r: usize) -> bool {
            let length = xs.vec.len();
            let x = r % length + 1;

            let ecdf = Ecdf::new(&mut xs.vec).unwrap();
            let actual = ecdf.rank(x).unwrap();

            // Not equal because e.g. all ranks of [0, 0] are 0. So
            // rank(1) == 0 and value of 0 gives ecdf == 1.0.
            (x as f64 / length as f64) <= ecdf.value(actual)
        }

        assert!(prop(Samples { vec: vec![0, 0] }, 0));
        check(prop as fn(Samples, usize) -> bool);
    }

    #[test]
    fn single_and_multiple_use_rank_agree() {
        fn prop(mut xs: Samples, r: usize) -> bool {
            let length = xs.vec.len();
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();

            let x = r % length + 1;
            multiple_use.rank(x).unwrap() == rank(&xs.vec, x).unwrap()
        }

        check(prop as fn(Samples, usize) -> bool);
    }

    #[test]
    fn min_is_leq_all_samples() {
        fn prop(mut xs: Samples) -> bool {
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();
            let actual = multiple_use.min();

            xs.vec.iter().all(|&x| actual <= x)
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn max_is_geq_all_samples() {
        fn prop(mut xs: Samples) -> bool {
            let multiple_use = Ecdf::new(&mut xs.vec).unwrap();
            let actual = multiple_use.max();

            xs.vec.iter().all(|&x| actual >= x)
        }

        check(prop as fn(Samples) -> bool);
    }
}
