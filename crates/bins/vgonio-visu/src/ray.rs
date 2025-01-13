use jabr::{Pnt3, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub org: Pnt3,
    pub dir: Vec3,
}

impl Ray {
    pub const fn new(org: Pnt3, dir: Vec3) -> Self { Ray { org, dir } }

    pub const fn empty() -> Self { Ray::new(Pnt3::zeros(), Vec3::zeros()) }

    pub fn at(&self, t: f64) -> Pnt3 { self.org + t * self.dir }
}
