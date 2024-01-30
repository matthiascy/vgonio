use jabr::{Clr3, Pnt3, Vec3};
use std::sync::atomic::AtomicU32;

use crate::{hit::HittableList, ray::Ray};

pub struct Camera {
    /// Image plane width in pixels.
    pub img_w: u32,
    /// Image plane height in pixels.
    pub img_h: u32,
    /// Image plane width to height ratio.
    pub ratio: f64,
    /// Camera origin.
    origin: Pnt3,
    /// Top left corner of the image plane.
    pixel_tlc: Pnt3,
    /// Offset to the next pixel in the u direction. (horizontal)
    pixel_delta_u: Vec3,
    /// Offset to the next pixel in the v direction. (vertical)
    pixel_delta_v: Vec3,
}

impl Camera {
    pub fn new(img_w: u32, img_h: u32) -> Self {
        let ratio = img_w as f64 / img_h as f64;
        let viewport_h = 2.0; // [-1, 1]
        let viewport_w = ratio * viewport_h;
        let focal_length = 1.0;
        let center = Pnt3::new(0.0, 0.0, 0.0);

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
        let viewport_u_axis = Vec3::new(viewport_w, 0.0, 0.0);
        let viewport_v_axis = Vec3::new(0.0, 0.0, -viewport_h);
        let pixel_delta_u = viewport_u_axis / img_w as f64;
        let pixel_delta_v = viewport_v_axis / img_h as f64;

        // Calculate viewport upper left corner in world/camera coordinates
        let viewport_upper_left_corner = center - 0.5 * viewport_u_axis - 0.5 * viewport_v_axis
            + Vec3::new(0.0, focal_length, 0.0);
        let pixel_tlc = viewport_upper_left_corner + 0.5 * pixel_delta_u + 0.5 * pixel_delta_v;

        Camera {
            img_w,
            img_h,
            ratio,
            origin: center,
            pixel_tlc,
            pixel_delta_u,
            pixel_delta_v,
        }
    }

    pub fn render(&self, world: &HittableList, spp: u32) {
        // let mut remaining_rows = AtomicU32::new(self.img_h);

        let mut image = image::RgbaImage::new(self.img_w, self.img_h);
        let mut rays = vec![Ray::empty(); spp as usize];
        let scale = 1.0 / spp as f64;
        for j in 0..self.img_h {
            print!("\rScanlines remaining: {}", self.img_h - j - 1);
            for i in 0..self.img_w {
                self.generate_rays(i, j, &mut rays);
                let mut pixel_color = Clr3::zeros();
                for ray in &rays {
                    pixel_color += ray_color(ray, world) * scale;
                }
                pixel_color = pixel_color.clamp(0.0, 0.999);
                image.put_pixel(
                    i,
                    j,
                    image::Rgba([
                        (256.0 * pixel_color.x) as u8,
                        (256.0 * pixel_color.y) as u8,
                        (256.0 * pixel_color.z) as u8,
                        255,
                    ]),
                );
            }
        }
        let filename = format!("image-{}x{}-{}spp.png", self.img_w, self.img_h, spp);
        image
            .save_with_format(filename, image::ImageFormat::Png)
            .unwrap();
        println!("\rDone.                   ");
    }

    /// Generates `n` rays for the pixel at `(u, v)`.
    fn generate_rays(&self, u: u32, v: u32, rays: &mut [Ray]) {
        let n = rays.len();
        let pixel_center = self.pixel_tlc
            + (u as f64 + 0.5) * self.pixel_delta_u
            + (v as f64 + 0.5) * self.pixel_delta_v;
        let mut samples = vec![Pnt3::new(0.0, 0.0, 0.0); n];

        if n > 1 {
            crate::random::samples_in_unit_square_2d(&mut samples);
        }

        for (ray, sample) in rays.iter_mut().zip(samples) {
            let pixel = pixel_center
                + (sample.x - 0.5) * self.pixel_delta_u
                + (sample.y - 0.5) * self.pixel_delta_v;
            let ray_dir = pixel - self.origin;
            *ray = Ray::new(self.origin, ray_dir);
        }
    }
}

fn ray_color(ray: &Ray, world: &HittableList) -> Clr3 {
    if let Some(rec) = world.hit(ray, 0.0..=f64::INFINITY) {
        return 0.5 * (rec.n + Clr3::new(1.0, 1.0, 1.0));
    }

    let unit_direction = ray.dir.normalize();
    let t = 0.5 * (unit_direction.z + 1.0);
    (1.0 - t) * Clr3::new(1.0, 1.0, 1.0) + t * Clr3::new(0.5, 0.7, 1.0)
}
