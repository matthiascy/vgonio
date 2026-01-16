// TODO: bring back the const methods
/// Forward array methods to upper wrapper.
macro_rules! forward_array_core_methods {
    ($($method:ident -> $rty:ty, #[$doc:meta]);*;) => {
        $(
            #[$doc]
            #[inline]
            pub fn $method(&self) -> $rty {
                self.0.$method()
            }
        )*
    };
    // (@const $($method:ident -> $rty:ty, #[$doc:meta]);*;) => {
    //     $(
    //         #[$doc]
    //         #[inline]
    //         pub const fn $method(&self) -> $rty {
    //             self.0.$method()
    //         }
    //     )*
    // };
}

/// Common array methods shared by all array types.
macro_rules! forward_array_core_common_methods {
    () => {
        forward_array_core_methods!(//@const
            shape -> &[usize], #[doc = "Returns the shape of the array."];
            strides -> &[usize], #[doc = "Returns the strides of the array."];
            order -> MemLayout, #[doc = "Returns the layout of the array."];
            dimension -> usize, #[doc = "Returns the number of dimensions of the array."];
            len -> usize, #[doc = "Returns the total number of elements in the array."];
        );
    }
}

mod arr_d;
mod arr_dy;
mod arr_dyn;
mod arr_s;
mod core;
mod dim;
mod mem;
mod ops;
mod packet;
pub mod shape;
mod traits;

pub use arr_d::*;
pub use arr_dy::*;
pub use arr_dyn::*;
pub use arr_s::*;
pub use mem::MemLayout;
pub use shape::{s, ConstShape};
