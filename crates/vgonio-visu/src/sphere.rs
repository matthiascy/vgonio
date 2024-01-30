use crate::{
    hit::{Hit, Hittable},
    ray::Ray,
};
use jabr::Pnt3;
use std::ops::RangeInclusive;

pub struct Sphere {
    pub c: Pnt3,
    pub r: f64,
}

impl Sphere {
    pub fn new(c: Pnt3, r: f64) -> Self { Sphere { c, r } }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t: RangeInclusive<f64>) -> Option<Hit> {
        let oc = ray.org - self.c;
        let a = ray.dir.norm_sqr();
        let half_b = oc.dot(&ray.dir);
        let c = oc.norm_sqr() - self.r * self.r;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();
        let mut root = (-half_b - sqrtd) / a;

        if !t.contains(&root) {
            root = (-half_b + sqrtd) / a;
            if !t.contains(&root) {
                return None;
            }
        }

        let p = ray.at(root);
        let n = (p - self.c).normalize();
        Some(Hit::new(ray, root, &n))
    }
}
