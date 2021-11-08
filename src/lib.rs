pub mod ecdf;
pub mod test;

pub use ecdf::{ecdf, percentile, permille, rank, Ecdf};
pub use test::{calculate_critical_value, test};
