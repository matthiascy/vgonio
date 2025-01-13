use crate::{material::Material, ray::Ray};
use jabr::{Pnt3, Vec3};
use std::{
    ops::RangeInclusive,
    sync::{Arc, Weak},
};

pub struct Hit {
    pub n: Vec3,
    pub p: Pnt3,
    pub t: f64,
    pub mat: Weak<dyn Material>,
}

impl Hit {
    /// Creates a new hit point.
    ///
    /// # Arguments
    ///
    /// * `ray` - The incident ray.
    /// * `t` - The distance from the ray origin to the hit point.
    /// * `n` - The outward normal vector on the surface of the hit point.
    pub fn new(ray: &Ray, t: f64, n: &Vec3, mat: Arc<dyn Material>) -> Self {
        let p = ray.at(t);
        Hit {
            n: *n,
            p,
            t,
            mat: Arc::downgrade(&mat),
        }
    }

    /// Returns if the ray is outside the surface.
    #[must_use]
    #[inline(always)]
    pub fn is_outside(&self, ray: &Ray) -> bool { ray.dir.dot(&self.n) < 0.0 }

    /// Returns if the ray is inside the surface.
    #[must_use]
    #[inline(always)]
    pub fn is_inside(&self, ray: &Ray) -> bool { ray.dir.dot(&self.n) > 0.0 }
}

pub trait Hittable: Send + Sync {
    fn hit(&self, ray: &Ray, t: RangeInclusive<f64>) -> Option<Hit>;
}

#[derive(Default)]
pub struct HittableList {
    objects: Vec<Arc<dyn Hittable>>,
}

impl HittableList {
    pub fn add(&mut self, object: Arc<dyn Hittable>) { self.objects.push(object); }

    pub fn clear(&mut self) { self.objects.clear(); }

    pub fn hit(&self, ray: &Ray, t: RangeInclusive<f64>) -> Option<Hit> {
        let mut closest_so_far = *t.end();
        let mut hit = None;

        for object in &self.objects {
            if let Some(temp_hit) = object.hit(ray, *t.start()..=closest_so_far) {
                closest_so_far = temp_hit.t;
                hit = Some(temp_hit);
            }
        }
        hit
    }
}
