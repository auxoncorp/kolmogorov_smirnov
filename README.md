# Overview
This is a fork of the [https://github.com/daithiocrualaoich/kolmogorov_smirnov](https://github.com/daithiocrualaoich/kolmogorov_smirnov)
library by [Daithi O Crualaoich](https://github.com/daithiocrualaoich). The fork is
motivated by a need for a non-panic'ing, minimally-allocating Kolmogorov-Smirnov test
in Rust.

# Fork Changes
----------------
### Added 
* Non-allocating variations on certain functions are provided (TODO - list these)
 
### Changed
* The primary `test` function returns a `Result<_, _>` rather than panicking
* The struct `TestResult` has been renamed to `TestOutcome` to reduce confusion with `Result`.
* Rust edition has been bumped to 2018
* The ECDF type no longer allocates on `new`

### Removed
* The detailed explanation and examples in the original crate's extended documentation [here](http://daithiocrualaoich.github.io/kolmogorov_smirnov).
* Docker support
* Several small command line utilities for KS-related functions
* Sample data and charts
* The `test_f64` function. Users may instead wrap their
floats using the [ordered-float](https://crates.io/crates/ordered-float) crate.

License
-------

    Copyright [2015] [Daithi O Crualaoich]

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
