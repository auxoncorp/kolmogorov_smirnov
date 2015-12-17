extern crate rand;

use std::env;
use rand::distributions::{Normal, IndependentSample};

/// Prints a sequence of Normal deviates.
///
/// # Examples
///
/// ```bash
/// cargo run --bin normal <num_deviates> <mean> <variance>
/// ```
/// This will print `<num_deviates>` float point numbers, one per line, to
/// standard output. These numbers will have a Normal distribution with the
/// specified mean and variance.
///
/// `<num_deviates>` must be a positive integer, `<mean>` and `<variance>` may
/// be integers or floating point numbers but `<variance>` must be positive.
fn main() {
  let args: Vec<String> = env::args().collect();

  let n: u32 = args[1].parse::<u32>().unwrap();
  let mean: f64 = args[2].parse::<f64>().unwrap();
  let variance: f64 = args[3].parse::<f64>().unwrap();

  let mut rng = rand::thread_rng();
  let normal = Normal::new(mean, variance.sqrt());

  for _ in 0..n {
    let x = normal.ind_sample(&mut rng);
    println!("{}", x)
  }
}
