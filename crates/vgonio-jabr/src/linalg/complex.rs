use crate::array::{s, Arr, MemLayout};
use num_traits::Num;

pub struct Complex<T: Num>(Arr<T, s![2], { MemLayout::ColMajor }>);
