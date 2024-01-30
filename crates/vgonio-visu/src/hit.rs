use crate::ray::Ray;
use jabr::{Pnt3, Vec3};
use std::{ops::RangeInclusive, sync::Arc};

pub struct Hit {
    pub n: Vec3,
    pub p: Pnt3,
    pub t: f64,
    pub front_face: bool,
}

impl Hit {
    /// Creates a new hit point.
    ///
    /// # Arguments
    ///
    /// * `ray` - The incident ray.
    /// * `t` - The distance from the ray origin to the hit point.
    /// * `n` - The outward normal vector on the surface of the hit point.
    pub fn new(ray: &Ray, t: f64, n: &Vec3) -> Self {
        let front_face = ray.dir.dot(n) < 0.0;
        let p = ray.at(t);
        Hit {
            n: *n,
            p,
            t,
            front_face,
        }
    }
}

pub trait Hittable: Send + Sync {
    fn hit(&self, ray: &Ray, t: RangeInclusive<f64>) -> Option<Hit>;
}

pub struct HittableList {
    objects: Vec<Arc<dyn Hittable>>,
}

impl HittableList {
    pub fn new() -> Self {
        HittableList {
            objects: Vec::new(),
        }
    }

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
