/// Forward array methods to upper wrapper.
macro forward_core_array_methods {
    ($($method:ident -> $rty:ty, #[$doc:meta]);*;) => {
        $(
            #[$doc]
            #[inline]
            pub fn $method(&self) -> $rty {
                self.0.$method()
            }
        )*
    },
    (@const $($method:ident -> $rty:ty, #[$doc:meta]);*;) => {
        $(
            #[$doc]
            #[inline]
            pub const fn $method(&self) -> $rty {
                self.0.$method()
            }
        )*
    },
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
pub use arr_dy::*;
pub use arr_dyn::*;
pub use arr_s::*;
pub use mem::MemLayout;
pub use shape::{s, ConstShape};
