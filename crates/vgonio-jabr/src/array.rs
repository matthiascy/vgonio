/// Forward array methods to the underlying array.
macro _forward_core_array_methods {
    ($($method:ident, #[$doc:meta]);*;) => {
        $(
            #[$doc]
            #[inline]
            pub fn $method(&self) -> &[usize] {
                self.0.$method()
            }
        )*
    },
    (@const $($method:ident, #[$doc:meta]);*;) => {
        $(
            #[$doc]
            #[inline]
            pub const fn $method(&self) -> &[usize] {
                self.0.$method()
            }
        )*
    },
}

macro forward_const_core_array_methods() {
    _forward_core_array_methods!(@const
        shape, #[doc = "Returns the shape of the array."];
        strides, #[doc = "Returns the strides of the array."];
        order, #[doc = "Returns the layout of the array."];
        dimension, #[doc = "Returns the number of dimensions of the array."];
    )
}

mod arr_d;
mod arr_dy;
mod arr_dyn;
mod arr_s;
mod core;
mod dim;
mod mem;
mod ops;
mod shape;

pub use arr_d::*;
pub use arr_dyn::*;
pub use arr_s::*;
pub use mem::MemLayout;
pub use shape::{s, ConstShape};
