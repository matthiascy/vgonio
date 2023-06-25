// Enable _mm_rcp14_ss
#![feature(stdsimd)]
// Enable macro 2.0
#![feature(decl_macro)]
// Enable const trait implementation
#![feature(const_trait_impl)]
// Enable const fn floating point arithmetic
#![feature(const_fn_floating_point_arithmetic)]
// Enable const mut references
#![feature(const_mut_refs)]
#![warn(missing_docs)]

pub mod error;
pub mod io;
pub mod math;
pub mod units;