use crate::array::{s, Arr, MemLayout};

pub struct Complex<T>(Arr<T, s![2], { MemLayout::ColMajor }>);
