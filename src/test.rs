//! Two Sample Kolmogorov-Smirnov Test

use std::cmp::{min, Ord};

/// Two sample test result.
#[derive(Debug)]
pub struct TestOutcome {
    pub is_rejected: bool,
    pub statistic: f64,
    pub reject_probability: f64,
    pub critical_value: f64,
    pub confidence: f64,
}

#[derive(Debug, PartialEq)]
pub enum TestError {
    /// The algorithm only supports comparing samples of size > 7
    SequenceHasFewerThanEightElements,
    /// The confidence value must be expressed as between 0.0 and 1.0
    /// E.G. 0.95
    ConfidenceMustBeBetweenZeroAndOneExclusive,
    /// The algorithm for estimating the critical value could not converge
    /// for the specified combination of sample sizes and confidence level.
    CouldNotConvergeOnCriticalValue {
        sample_size_1: usize,
        sample_size_2: usize,
        confidence: f64,
    },
    /// Could not converge on Kolmogorov-Smirnov probability function
    CouldNotConvergeOnKolmogorovSmirnovProbabilityFunction { lambda: f64 },
    /// Our algorithm for determining a reject probability produced an out-of-bounds value.
    InvalidEstimatedKolmogorovSmirnovProbabilityFunctionValue { lambda: f64, value: f64 },
}

/// Perform a two sample Kolmogorov-Smirnov test on given samples.
///
/// The samples must have length > 7 elements for the test to be valid.
/// The confidence must be greater than 0.0 and less than 1.0
///
/// Mutates the order of the input samples, but does not allocate.
///
/// # Examples
///
/// ```
/// extern crate kolmogorov_smirnov as ks;
///
/// let xs = vec!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
/// let ys = vec!(12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
/// let confidence = 0.95;
///
/// let result = ks::test(&xs, &ys, confidence).unwrap();
///
/// if result.is_rejected {
///     println!("{:?} and {:?} are not from the same distribution with probability {}.",
///       xs, ys, result.reject_probability);
/// }
/// ```
pub fn test_nonallocating<T: Ord + Clone>(
    xs: &mut [T],
    ys: &mut [T],
    confidence: f64,
) -> Result<TestOutcome, TestError> {
    if xs.len() < 8 || ys.len() < 8 {
        return Err(TestError::SequenceHasFewerThanEightElements);
    }
    if confidence.is_nan() || !(0.0 < confidence && confidence < 1.0) {
        return Err(TestError::ConfidenceMustBeBetweenZeroAndOneExclusive);
    }

    let statistic = calculate_statistic_nonallocating(xs, ys)?;
    let critical_value = calculate_critical_value(xs.len(), ys.len(), confidence)?;

    let reject_probability = calculate_reject_probability(statistic, xs.len(), ys.len())?;
    let is_rejected = reject_probability > confidence;

    Ok(TestOutcome {
        is_rejected,
        statistic,
        reject_probability,
        critical_value,
        confidence,
    })
}

/// Perform a two sample Kolmogorov-Smirnov test on given samples.
///
/// The samples must have length > 7 elements for the test to be valid.
/// The confidence must be greater than 0.0 and less than 1.0
///
/// # Examples
///
/// ```
/// extern crate kolmogorov_smirnov as ks;
///
/// let xs = vec!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
/// let ys = vec!(12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
/// let confidence = 0.95;
///
/// let result = ks::test(&xs, &ys, confidence).unwrap();
///
/// if result.is_rejected {
///     println!("{:?} and {:?} are not from the same distribution with probability {}.",
///       xs, ys, result.reject_probability);
/// }
/// ```
pub fn test<T: Ord + Clone>(xs: &[T], ys: &[T], confidence: f64) -> Result<TestOutcome, TestError> {
    if xs.len() < 8 || ys.len() < 8 {
        return Err(TestError::SequenceHasFewerThanEightElements);
    }
    if confidence.is_nan() || !(0.0 < confidence && confidence < 1.0) {
        return Err(TestError::ConfidenceMustBeBetweenZeroAndOneExclusive);
    }

    let statistic = calculate_statistic_allocating(xs, ys)?;
    let critical_value = calculate_critical_value(xs.len(), ys.len(), confidence)?;

    let reject_probability = calculate_reject_probability(statistic, xs.len(), ys.len())?;
    let is_rejected = reject_probability > confidence;

    Ok(TestOutcome {
        is_rejected,
        statistic,
        reject_probability,
        critical_value,
        confidence,
    })
}

/// Calculate the test statistic for the two sample Kolmogorov-Smirnov test.
///
/// The test statistic is the maximum vertical distance between the ECDFs of
/// the two samples.
///
/// This function sorts the input samples.
fn calculate_statistic_nonallocating<T: Ord + Clone>(
    xs: &mut [T],
    ys: &mut [T],
) -> Result<f64, TestError> {
    if xs.is_empty() || ys.is_empty() {
        return Err(TestError::SequenceHasFewerThanEightElements);
    }

    xs.sort();
    ys.sort();

    Ok(calculate_statistic_inner_presorted(xs, ys))
}
/// Calculate the test statistic for the two sample Kolmogorov-Smirnov test.
///
/// The test statistic is the maximum vertical distance between the ECDFs of
/// the two samples.
///
/// This function copies each input into a new Vec in order to do sorting.
fn calculate_statistic_allocating<T: Ord + Clone>(xs: &[T], ys: &[T]) -> Result<f64, TestError> {
    if xs.is_empty() || ys.is_empty() {
        return Err(TestError::SequenceHasFewerThanEightElements);
    }

    let (xs, ys) = {
        let mut xs = xs.to_vec();
        let mut ys = ys.to_vec();

        // xs and ys must be sorted for the stepwise ECDF calculations to work.
        xs.sort();
        ys.sort();
        (xs, ys)
    };
    Ok(calculate_statistic_inner_presorted(&xs, &ys))
}
/// Internal use only, because it strongly assumes that the input samples
/// are pre-sorted and of an acceptable length (e.g. > 7)
///
/// Calculate the test statistic for the two sample Kolmogorov-Smirnov test.
///
/// The test statistic is the maximum vertical distance between the ECDFs of
/// the two samples.
fn calculate_statistic_inner_presorted<T: std::cmp::PartialEq + Ord>(xs: &[T], ys: &[T]) -> f64 {
    // The current value testing for ECDF difference. Sweeps up through elements
    // present in xs and ys.
    let mut current: &T;

    // i, j index the first values in xs and ys that are greater than current.
    let mut i = 0;
    let mut j = 0;

    // ecdf_xs, ecdf_ys always hold the ECDF(current) of xs and ys.
    let mut ecdf_xs = 0.0;
    let mut ecdf_ys = 0.0;

    // The test statistic value computed over values <= current.
    let mut statistic = 0.0;

    let n = xs.len();
    let m = ys.len();
    while i < n && j < m {
        // Advance i through duplicate samples in xs.
        let x_i = &xs[i];
        while i + 1 < n && *x_i == xs[i + 1] {
            i += 1;
        }

        // Advance j through duplicate samples in ys.
        let y_j = &ys[j];
        while j + 1 < m && *y_j == ys[j + 1] {
            j += 1;
        }

        // Step to the next sample value in the ECDF sweep from low to high.
        current = min(x_i, y_j);

        // Update invariant conditions for i, j, ecdf_xs, and ecdf_ys.
        if current == x_i {
            ecdf_xs = (i + 1) as f64 / n as f64;
            i += 1;
        }
        if current == y_j {
            ecdf_ys = (j + 1) as f64 / m as f64;
            j += 1;
        }

        // Update invariant conditions for the test statistic.
        let diff = (ecdf_xs - ecdf_ys).abs();
        if diff > statistic {
            statistic = diff;
        }
    }

    // Don't need to walk the rest of the samples because one of the ecdfs is
    // already one and the other will be increasing up to one. This means the
    // difference will be monotonically decreasing, so we have our test
    // statistic value already.

    statistic
}

/// Calculate the probability that the null hypothesis is false for a two sample
/// Kolmogorov-Smirnov test. Can only reject the null hypothesis if this
/// evidence exceeds the confidence level required.
fn calculate_reject_probability(statistic: f64, n1: usize, n2: usize) -> Result<f64, TestError> {
    // Only supports samples of size > 7.
    if n1 < 8 || n2 < 8 {
        return Err(TestError::SequenceHasFewerThanEightElements);
    }

    let n1 = n1 as f64;
    let n2 = n2 as f64;

    let factor = ((n1 * n2) / (n1 + n2)).sqrt();
    let term = (factor + 0.12 + 0.11 / factor) * statistic;

    let reject_probability = 1.0 - probability_kolmogorov_smirnov(term)?;
    if !(0.0..=1.0).contains(&reject_probability) {
        return Err(
            TestError::InvalidEstimatedKolmogorovSmirnovProbabilityFunctionValue {
                lambda: term,
                value: reject_probability,
            },
        );
    }
    Ok(reject_probability)
}

/// Calculate the critical value for the two sample Kolmogorov-Smirnov test.
///
///
/// No convergence error returned if the binary search does not locate the critical
/// value in less than 200 iterations.
///
/// # Examples
///
/// ```
/// extern crate kolmogorov_smirnov as ks;
///
/// let critical_value = ks::calculate_critical_value(256, 256, 0.95).unwrap();
/// println!("Critical value at 95% confidence for samples of size 256 is {}",
///       critical_value);
/// ```
pub fn calculate_critical_value(n1: usize, n2: usize, confidence: f64) -> Result<f64, TestError> {
    if confidence.is_nan() || !(0.0 < confidence && confidence < 1.0) {
        return Err(TestError::ConfidenceMustBeBetweenZeroAndOneExclusive);
    }

    // Only supports samples of size > 7.
    if n1 < 8 || n2 < 8 {
        return Err(TestError::SequenceHasFewerThanEightElements);
    }

    // The test statistic is between zero and one so can binary search quickly
    // for the critical value.
    let mut low = 0.0;
    let mut high = 1.0;

    for _ in 1..200 {
        if low + 1e-8 >= high {
            return Ok(high);
        }

        let mid = low + (high - low) / 2.0;
        let reject_probability = calculate_reject_probability(mid, n1, n2)?;

        if reject_probability > confidence {
            // Maintain invariant that reject_probability(high) > confidence.
            high = mid;
        } else {
            // Maintain invariant that reject_probability(low) <= confidence.
            low = mid;
        }
    }
    Err(TestError::CouldNotConvergeOnCriticalValue {
        sample_size_1: n1,
        sample_size_2: n2,
        confidence,
    })
}

/// Calculate the Kolmogorov-Smirnov probability function.
fn probability_kolmogorov_smirnov(lambda: f64) -> Result<f64, TestError> {
    if lambda == 0.0 {
        return Ok(1.0);
    }

    let minus_two_lambda_squared = -2.0 * lambda * lambda;
    let mut q_ks = 0.0;

    for j in 1..200 {
        let sign = if j % 2 == 1 { 1.0 } else { -1.0 };

        let j = j as f64;
        let term = sign * 2.0 * (minus_two_lambda_squared * j * j).exp();

        q_ks += term;

        if term.abs() < 1e-8 {
            // Trim results that exceed 1.
            return Ok(q_ks.min(1.0));
        }
    }
    Err(TestError::CouldNotConvergeOnKolmogorovSmirnovProbabilityFunction { lambda })
}

#[cfg(test)]
mod tests {
    extern crate quickcheck;
    extern crate rand;

    use self::quickcheck::{Arbitrary, Gen, QuickCheck, StdGen, Testable};
    use self::rand::Rng;
    use std::cmp;
    use std::usize;

    use super::{test, TestError};
    use crate::ecdf::Ecdf;

    const EPSILON: f64 = 1e-10;

    fn check<A: Testable>(f: A) {
        // Need - 1 to ensure space for creating non-overlapping samples.
        let g = StdGen::new(rand::thread_rng(), usize::MAX - 1);
        QuickCheck::new().gen(g).quickcheck(f);
    }

    /// Wrapper for generating sample data with QuickCheck.
    ///
    /// Samples must be sequences of u64 values with more than 7 elements.
    #[derive(Debug, Clone)]
    struct Samples {
        vec: Vec<u64>,
    }

    impl Samples {
        fn min(&self) -> u64 {
            let &min = self.vec.iter().min().unwrap();
            min
        }

        fn max(&self) -> u64 {
            let &max = self.vec.iter().max().unwrap();
            max
        }

        fn shuffle(&mut self) {
            let mut rng = rand::thread_rng();
            rng.shuffle(&mut self.vec);
        }
    }

    impl Arbitrary for Samples {
        fn arbitrary<G: Gen>(g: &mut G) -> Samples {
            // Limit size of generated sample set to 1024
            let max = cmp::min(g.size(), 1024);

            let size = g.gen_range(8, max);
            let vec = (0..size).map(|_| u64::arbitrary(g)).collect();

            Samples { vec: vec }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Samples>> {
            let vec: Vec<u64> = self.vec.clone();
            let shrunk: Box<dyn Iterator<Item = Vec<u64>>> = vec.shrink();

            Box::new(shrunk.filter(|v| v.len() > 7).map(|v| Samples { vec: v }))
        }
    }

    #[test]
    fn test_error_on_empty_samples_set() {
        let xs: Vec<u64> = vec![];
        let ys: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        assert_eq!(
            TestError::SequenceHasFewerThanEightElements,
            test(&xs, &ys, 0.95).unwrap_err()
        );
    }

    #[test]
    fn test_error_on_empty_other_samples_set() {
        let xs: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let ys: Vec<u64> = vec![];
        assert_eq!(
            TestError::SequenceHasFewerThanEightElements,
            test(&xs, &ys, 0.95).unwrap_err()
        );
    }

    #[test]
    fn test_error_on_confidence_leq_zero() {
        let xs: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let ys: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        assert_eq!(
            TestError::ConfidenceMustBeBetweenZeroAndOneExclusive,
            test(&xs, &ys, 0.0).unwrap_err()
        );
    }

    #[test]
    fn test_error_on_confidence_geq_one() {
        let xs: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let ys: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        assert_eq!(
            TestError::ConfidenceMustBeBetweenZeroAndOneExclusive,
            test(&xs, &ys, 1.0).unwrap_err()
        );
    }

    /// Alternative calculation for the test statistic for the two sample
    /// Kolmogorov-Smirnov test. This simple implementation is used as a
    /// verification check against actual calculation used.
    fn calculate_statistic_alt<T: Ord + Clone>(xs: &[T], ys: &[T]) -> f64 {
        assert!(xs.len() > 0 && ys.len() > 0);
        let mut xs_alt = xs.to_vec();
        let mut ys_alt = ys.to_vec();

        let ecdf_xs = Ecdf::new(&mut xs_alt).unwrap();
        let ecdf_ys = Ecdf::new(&mut ys_alt).unwrap();

        let mut statistic = 0.0;

        for x in xs.iter() {
            let diff = (ecdf_xs.value(x.clone()) - ecdf_ys.value(x.clone())).abs();
            if diff > statistic {
                statistic = diff;
            }
        }

        for y in ys.iter() {
            let diff = (ecdf_xs.value(y.clone()) - ecdf_ys.value(y.clone())).abs();
            if diff > statistic {
                statistic = diff;
            }
        }

        statistic
    }

    #[test]
    fn test_calculate_statistic() {
        fn prop(xs: Samples, ys: Samples) -> bool {
            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();
            let actual = result.statistic;
            let expected = calculate_statistic_alt(&xs.vec, &ys.vec);

            actual == expected
        }

        check(prop as fn(Samples, Samples) -> bool);
    }

    #[test]
    fn test_statistic_is_between_zero_and_one() {
        fn prop(xs: Samples, ys: Samples) -> bool {
            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();
            let actual = result.statistic;

            0.0 <= actual && actual <= 1.0
        }

        check(prop as fn(Samples, Samples) -> bool);
    }

    #[test]
    fn test_statistic_is_zero_for_identical_samples() {
        fn prop(xs: Samples) -> bool {
            let ys = xs.clone();

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();

            result.statistic == 0.0
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn test_statistic_is_zero_for_permuted_sample() {
        fn prop(xs: Samples) -> bool {
            let mut ys = xs.clone();
            ys.shuffle();

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();

            result.statistic == 0.0
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn test_statistic_is_one_for_samples_with_no_overlap_in_support() {
        fn prop(xs: Samples) -> bool {
            let mut ys = xs.clone();

            // Shift ys so that ys.min > xs.max.
            let ys_min = xs.max() + 1;
            ys.vec = ys.vec.iter().map(|&y| cmp::max(y, ys_min)).collect();

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();

            result.statistic == 1.0
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn test_statistic_is_one_half_for_sample_with_non_overlapping_in_support_replicate_added() {
        fn prop(xs: Samples) -> bool {
            let mut ys = xs.clone();

            // Shift ys so that ys.min > xs.max.
            let ys_min = xs.max() + 1;
            ys.vec = ys.vec.iter().map(|&y| cmp::max(y, ys_min)).collect();

            // Add all the original items back too.
            for &x in xs.vec.iter() {
                ys.vec.push(x);
            }

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();

            result.statistic == 0.5
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn test_statistic_is_one_div_length_for_sample_with_additional_low_value() {
        fn prop(xs: Samples) -> bool {
            // Add a extra sample of early weight to ys.
            let min = xs.min();
            let mut ys = xs.clone();
            ys.vec.push(min - 1);

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();
            let expected = 1.0 / ys.vec.len() as f64;

            result.statistic == expected
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn test_statistic_is_one_div_length_for_sample_with_additional_high_value() {
        fn prop(xs: Samples) -> bool {
            // Add a extra sample of late weight to ys.
            let max = xs.max();
            let mut ys = xs.clone();
            ys.vec.push(max + 1);

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();
            let expected = 1.0 / ys.vec.len() as f64;

            (result.statistic - expected).abs() < EPSILON
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn test_statistic_is_one_div_length_for_sample_with_additional_low_and_high_values() {
        fn prop(xs: Samples) -> bool {
            // Add a extra sample of late weight to ys.
            let min = xs.min();
            let max = xs.max();

            let mut ys = xs.clone();

            ys.vec.push(min - 1);
            ys.vec.push(max + 1);

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();
            let expected = 1.0 / ys.vec.len() as f64;

            (result.statistic - expected).abs() < EPSILON
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn test_statistic_is_n_div_length_for_sample_with_additional_n_low_values() {
        fn prop(xs: Samples, n: u8) -> bool {
            // Add extra sample of early weight to ys.
            let min = xs.min();
            let mut ys = xs.clone();
            for j in 0..n {
                ys.vec.push(min - (j as u64) - 1);
            }

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();
            let expected = n as f64 / ys.vec.len() as f64;

            result.statistic == expected
        }

        check(prop as fn(Samples, u8) -> bool);
    }

    #[test]
    fn test_statistic_is_n_div_length_for_sample_with_additional_n_high_values() {
        fn prop(xs: Samples, n: u8) -> bool {
            // Add extra sample of early weight to ys.
            let max = xs.max();
            let mut ys = xs.clone();
            for j in 0..n {
                ys.vec.push(max + (j as u64) + 1);
            }

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();
            let expected = n as f64 / ys.vec.len() as f64;

            (result.statistic - expected).abs() < EPSILON
        }

        check(prop as fn(Samples, u8) -> bool);
    }

    #[test]
    fn test_statistic_is_n_div_length_for_sample_with_additional_n_low_and_high_values() {
        fn prop(xs: Samples, n: u8) -> bool {
            // Add extra sample of early weight to ys.
            let min = xs.min();
            let max = xs.max();
            let mut ys = xs.clone();
            for j in 0..n {
                ys.vec.push(min - (j as u64) - 1);
                ys.vec.push(max + (j as u64) + 1);
            }

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();
            let expected = n as f64 / ys.vec.len() as f64;

            (result.statistic - expected).abs() < EPSILON
        }

        check(prop as fn(Samples, u8) -> bool);
    }

    #[test]
    fn test_statistic_is_n_or_m_div_length_for_sample_with_additional_n_low_and_m_high_values() {
        fn prop(xs: Samples, n: u8, m: u8) -> bool {
            // Add extra sample of early weight to ys.
            let min = xs.min();
            let max = xs.max();
            let mut ys = xs.clone();
            for j in 0..n {
                ys.vec.push(min - (j as u64) - 1);
            }

            for j in 0..m {
                ys.vec.push(max + (j as u64) + 1);
            }

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();
            let expected = cmp::max(n, m) as f64 / ys.vec.len() as f64;

            (result.statistic - expected).abs() < EPSILON
        }

        check(prop as fn(Samples, u8, u8) -> bool);
    }

    #[test]
    fn test_is_rejected_if_reject_probability_greater_than_confidence() {
        fn prop(xs: Samples, ys: Samples) -> bool {
            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();

            if result.is_rejected {
                result.reject_probability > 0.95
            } else {
                result.reject_probability <= 0.95
            }
        }

        check(prop as fn(Samples, Samples) -> bool);
    }

    #[test]
    fn test_reject_probability_is_zero_for_identical_samples() {
        fn prop(xs: Samples) -> bool {
            let ys = xs.clone();

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();

            result.reject_probability == 0.0
        }

        check(prop as fn(Samples) -> bool);
    }

    #[test]
    fn test_reject_probability_is_zero_for_permuted_sample() {
        fn prop(xs: Samples) -> bool {
            let mut ys = xs.clone();
            ys.shuffle();

            let result = test(&xs.vec, &ys.vec, 0.95).unwrap();

            result.reject_probability == 0.0
        }

        check(prop as fn(Samples) -> bool);
    }
}
