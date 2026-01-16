use crate::array::{s, Array, MemLayout};
use num_traits::Num;

pub struct Complex<T: Num>(Array<T, s![2], { MemLayout::ColMajor }>);
