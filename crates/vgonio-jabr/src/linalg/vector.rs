use crate::array::{s, Array, ConstShape, MemLayout};

pub struct Vector<T, const N: usize>(Array<T, s![N], { MemLayout::ColMajor }>)
where
    [(); <s![N] as ConstShape>::N_DIMS]:,
    [(); <s![N] as ConstShape>::N_ELEMS]:;

pub type Vec2<T> = Vector<T, 2>;
pub type Vec3<T> = Vector<T, 3>;
pub type Vec4<T> = Vector<T, 4>;

pub struct Point<T, const N: usize>(Array<T, s![N], { MemLayout::ColMajor }>)
where
    [(); <s![N] as ConstShape>::N_DIMS]:,
    [(); <s![N] as ConstShape>::N_ELEMS]:;

pub type Pnt2<T> = Point<T, 2>;
pub type Pnt3<T> = Point<T, 3>;
pub type Pnt4<T> = Point<T, 4>;

pub struct Normal<T, const N: usize>(Array<T, s![N], { MemLayout::ColMajor }>)
where
    [(); <s![N] as ConstShape>::N_DIMS]:,
    [(); <s![N] as ConstShape>::N_ELEMS]:;

pub type Nrm2<T> = Normal<T, 2>;
pub type Nrm3<T> = Normal<T, 3>;
