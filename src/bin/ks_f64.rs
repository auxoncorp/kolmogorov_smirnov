extern crate kolmogorov_smirnov as ks;

use std::cmp::Ordering;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn parse_float(s: String) -> f64 {
    s.parse::<f64>().expect("Not a floating point number.")
}

/// Runs a Kolmogorov-Smirnov test on floating point data files.
///
/// Input files must be single-column headerless data files. The data samples
/// are tested against each other at the 95% confidence level.
///
/// # Examples
///
/// ```bash
/// cargo run --bin ks_f64 <file1> <file2>
/// ```
///
/// This will print the test result to standard output.
fn main() {
    let args: Vec<String> = env::args().collect();

    let path1 = Path::new(&args[1]);
    let path2 = Path::new(&args[2]);

    let file1 = BufReader::new(File::open(&path1).unwrap());
    let file2 = BufReader::new(File::open(&path2).unwrap());

    let lines1 = file1.lines().map(|line| line.unwrap());
    let lines2 = file2.lines().map(|line| line.unwrap());

    let xs: Vec<OrderableF64> = lines1.map(parse_float).map(OrderableF64::new).collect();
    let ys: Vec<OrderableF64> = lines2.map(parse_float).map(OrderableF64::new).collect();

    let result = ks::test(&xs, &ys, 0.95).expect("Could not compute test value.");

    if result.is_rejected {
        println!("Samples are from different distributions.");
    } else {
        println!("Samples are from the same distribution.");
    }

    println!("test statistic = {}", result.statistic);
    println!("critical value = {}", result.critical_value);
    println!("reject probability = {}", result.reject_probability);
}

/// Wrapper type for f64 to implement Ord and make usable with test.
#[derive(PartialEq, Clone)]
struct OrderableF64 {
    val: f64,
}

impl OrderableF64 {
    fn new(val: f64) -> OrderableF64 {
        OrderableF64 { val }
    }
}

impl Eq for OrderableF64 {}

impl PartialOrd for OrderableF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl Ord for OrderableF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.val.partial_cmp(&other.val).unwrap()
    }
}
