use jabr::{Clr3, Pnt3, Vec3};

use crate::{hit::HittableList, random::random_point_in_unit_disk_xz, ray::Ray};

pub struct Camera {
    /// Image plane width to height ratio.
    pub ratio: f64,
    /// Vertical field of view in radians.
    pub vfov: f64,
    /// Camera origin.
    origin: Pnt3,
    /// Top left corner of the image plane.
    pixel_tlc: Pnt3,
    /// Offset to the next pixel in the u direction. (horizontal)
    pixel_delta_u: Vec3,
    /// Offset to the next pixel in the v direction. (vertical)
    pixel_delta_v: Vec3,
    /// Camera frame basis right vector - x/u axis.
    basis_right: Vec3,
    /// Camera frame basis forward vector - y/v axis.
    basis_forward: Vec3,
    /// Camera frame basis up vector - z/w axis.
    basis_up: Vec3,
    /// The defocus angle in radians.
    defocus_angle: f64,
    /// Basis vector for the defocus disk, in horizontal direction.
    defocus_disk_u: Vec3,
    /// Basis vector for the defocus disk, in vertical direction.
    defocus_disk_v: Vec3,
}

impl Camera {
    /// Creates a new camera.
    ///
    /// # Arguments
    ///
    /// * `origin` - Camera position.
    /// * `lookat` - Camera lookat point.
    /// * `up` - Camera up vector.
    /// * `img_w` - Image plane width in pixels.
    /// * `img_h` - Image plane height in pixels.
    /// * `vfov` - Vertical field of view in degrees.
    /// * `defocus_angle` - Variation angle of rays through each pixel.
    /// * `focus_distance` - Distance from camera origin to plane of perfect
    ///   focus.
    pub fn new(
        origin: Pnt3,
        lookat: Pnt3,
        up: Vec3,
        img_w: u32,
        img_h: u32,
        vfov: f64,
        defocus_angle: f64,
        focus_distance: f64,
    ) -> Self {
        let ratio = img_w as f64 / img_h as f64;
        let vfov = vfov.to_radians();
        let h = (vfov * 0.5).tan();

        // Determine viewport dimensions.
        // The distance from the camera center to the image plane.
        // let focal_length = (lookat - origin).norm();
        let viewport_h = 2.0 * h * focus_distance;
        let viewport_w = ratio * viewport_h;

        // Construct the camera frame basis.
        let forward = (lookat - origin).normalize();
        let right = forward.cross(&up.normalize()).normalize();
        let up = right.cross(&forward).normalize();

        // Camera/World coordinate system
        // Right-handed Z-up coordinate system
        // z    y
        // |  /
        // |/
        // ----> x

        // Viewport local coordinate system
        // ---> u
        // |
        // v
        let viewport_u_axis = viewport_w * right;
        let viewport_v_axis = -viewport_h * up;
        let pixel_delta_u = viewport_u_axis / img_w as f64;
        let pixel_delta_v = viewport_v_axis / img_h as f64;

        // Calculate viewport upper left corner in world/camera coordinates
        let viewport_upper_left_corner =
            // origin - 0.5 * viewport_u_axis - 0.5 * viewport_v_axis + focal_length * forward;
            origin - 0.5 * viewport_u_axis - 0.5 * viewport_v_axis + focus_distance * forward;
        let pixel_tlc = viewport_upper_left_corner + 0.5 * pixel_delta_u + 0.5 * pixel_delta_v;

        // Calculate the camera defocus disk basis vectors.
        let defocus_angle = defocus_angle.to_radians();
        let defocus_radius = focus_distance * (defocus_angle * 0.5).tan();
        let defocus_disk_u = right * defocus_radius;
        let defocus_disk_v = up * defocus_radius;

        Camera {
            ratio,
            vfov,
            origin,
            pixel_tlc,
            pixel_delta_u,
            pixel_delta_v,
            basis_forward: forward,
            basis_right: right,
            basis_up: up,
            defocus_angle,
            defocus_disk_u,
            defocus_disk_v,
        }
    }

    /// Generates `n` rays around the pixel at `(u, v)`.
    pub fn generate_rays(&self, u: u32, v: u32, rays: &mut [Ray]) {
        let n = rays.len();
        let pixel_center = self.pixel_tlc
            + (u as f64 + 0.5) * self.pixel_delta_u
            + (v as f64 + 0.5) * self.pixel_delta_v;
        let mut samples = vec![Pnt3::new(0.0, 0.0, 0.0); n].into_boxed_slice();

        if n > 1 {
            crate::random::samples_in_unit_square_2d(&mut samples);
        }

        let ray_origin = if self.defocus_angle <= 0.0 {
            self.origin
        } else {
            let pnt = random_point_in_unit_disk_xz();
            self.origin + (pnt.x * self.defocus_disk_u) + (pnt.z * self.defocus_disk_v)
        };

        for (ray, sample) in rays.iter_mut().zip(samples) {
            let pixel = pixel_center
                + (sample.x - 0.5) * self.pixel_delta_u
                + (sample.y - 0.5) * self.pixel_delta_v;
            let ray_dir = pixel - self.origin;
            *ray = Ray::new(ray_origin, ray_dir);
        }
    }
}

/// Recursivly computes the color of a ray after it hits the world.
///
/// # Arguments
///
/// * `ray` - The ray to compute the color for.
/// * `world` - The world containing all objects.
/// * `bounces` - The number of bounces the ray has made.
/// * `max_bounces` - The maximum number of bounces a ray can make.
///
/// # Returns
///
/// The color in linear RGB of the ray after it hits the world.
pub fn ray_color(ray: &Ray, world: &HittableList, bounces: u32, max_bounces: u32) -> Clr3 {
    if bounces >= max_bounces {
        return Clr3::zeros();
    }

    if let Some(rec) = world.hit(ray, 0.0001..=f64::INFINITY) {
        let mat = rec.mat.upgrade().unwrap();
        if let Some((scattered, attenuation)) = mat.scatter(ray, &rec) {
            return attenuation * ray_color(&scattered, world, bounces + 1, max_bounces);
        }
        return Clr3::zeros();
    }

    // Background color
    let unit_direction = ray.dir.normalize();
    let t = 0.5 * (unit_direction.z + 1.0);
    (1.0 - t) * Clr3::new(1.0, 1.0, 1.0) + t * Clr3::new(0.5, 0.7, 1.0)
}
