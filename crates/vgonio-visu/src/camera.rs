use jabr::{Clr3, Pnt3, Vec3};

use crate::{hit::HittableList, ray::Ray};

pub struct Camera {
    /// Image plane width in pixels.
    pub img_w: u32,
    /// Image plane height in pixels.
    pub img_h: u32,
    /// Image plane width to height ratio.
    pub ratio: f64,
    /// Camera origin.
    center: Pnt3,
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
            center,
            pixel_tlc,
            pixel_delta_u,
            pixel_delta_v,
        }
    }

    pub fn render(&self, world: &HittableList) {
        print!("P3\n{} {}\n255\n", self.img_w, self.img_h);

        for j in 0..self.img_h {
            eprint!("\rScanlines remaining: {}", self.img_h - j - 1);
            for i in 0..self.img_w {
                let pixel_center = self.pixel_tlc
                    + (i as f64 + 0.5) * self.pixel_delta_u
                    + (j as f64 + 0.5) * self.pixel_delta_v;
                let ray_dir = pixel_center - self.center;
                let ray = Ray::new(self.center, ray_dir);
                let pixel_color = ray_color(&ray, &world);
                write_color(&pixel_color);
            }
        }

        eprintln!("\rDone.                   ");
    }
}

fn write_color(pixel_color: &Clr3) {
    println!(
        "{} {} {}",
        (255.999 * pixel_color.x) as u8,
        (255.999 * pixel_color.y) as u8,
        (255.999 * pixel_color.z) as u8
    );
}

fn ray_color(ray: &Ray, world: &HittableList) -> Clr3 {
    if let Some(rec) = world.hit(ray, 0.0..=f64::INFINITY) {
        return 0.5 * (rec.n + Clr3::new(1.0, 1.0, 1.0));
    }

    let unit_direction = ray.dir.normalize();
    let t = 0.5 * (unit_direction.z + 1.0);
    (1.0 - t) * Clr3::new(1.0, 1.0, 1.0) + t * Clr3::new(0.5, 0.7, 1.0)
}
