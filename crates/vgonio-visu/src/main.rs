use clap::Parser;
use jabr::{Clr3, Vec3};
use std::env::args;
use vgonio_visu::ray::Ray;

fn write_color(pixel_color: &Clr3) {
    println!(
        "{} {} {}",
        (255.999 * pixel_color.x) as u8,
        (255.999 * pixel_color.y) as u8,
        (255.999 * pixel_color.z) as u8
    );
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
#[command(next_line_help = true)]
struct Args {
    #[arg(short = 'x', default_value_t = 256)]
    width: u32,
    #[arg(short = 'y', default_value_t = 256)]
    height: u32,
}

fn hit_sphere(center: &Vec3, radius: f64, ray: &Ray) -> bool {
    let oc = ray.org - center;
    let a = ray.dir.dot(&ray.dir);
    let b = 2.0 * oc.dot(&ray.dir);
    let c = oc.dot(&oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    discriminant >= 0.0
}

fn ray_color(ray: &Ray) -> Clr3 {
    if hit_sphere(&Vec3::new(0.0, 1.0, 0.0), 0.5, ray) {
        return Clr3::new(1.0, 0.0, 0.0);
    }

    let unit_direction = ray.dir.normalize();
    let t = 0.5 * (unit_direction.z + 1.0);
    (1.0 - t) * Clr3::new(1.0, 1.0, 1.0) + t * Clr3::new(0.5, 0.7, 1.0)
}

fn main() {
    let args = Args::parse();
    let (image_width, image_height) = (args.width, args.height);
    if args.width == 0 || args.height == 0 {
        eprintln!("Image dimensions must be positive integers.");
        return;
    }

    let ratio = image_width as f64 / image_height as f64;
    let viewport_height = 2.0; // [-1, 1]
    let viewport_width = ratio * viewport_height;
    let focal_length = 1.0;
    let camera_origin = Vec3::new(0.0, 0.0, 0.0);

    // Viewport local coordinate system
    // ---> u
    // |
    // v
    let viewport_u_axis = Vec3::new(viewport_width, 0.0, 0.0);
    let viewport_v_axis = Vec3::new(0.0, 0.0, -viewport_height);
    let pixel_delta_u = viewport_u_axis / image_width as f64;
    let pixel_delta_v = viewport_v_axis / image_height as f64;

    // Calculate viewport upper left corner in world/camera coordinates
    let viewport_upper_left_corner = camera_origin - 0.5 * viewport_u_axis - 0.5 * viewport_v_axis
        + Vec3::new(0.0, focal_length, 0.0);
    let pixel_top_left_corner =
        viewport_upper_left_corner + 0.5 * pixel_delta_u + 0.5 * pixel_delta_v;

    print!("P3\n{} {}\n255\n", image_width, image_height);

    for j in 0..image_height {
        eprint!("\rScanlines remaining: {}", image_height - j - 1);
        for i in 0..image_width {
            let pixel_center = pixel_top_left_corner
                + (i as f64 + 0.5) * pixel_delta_u
                + (j as f64 + 0.5) * pixel_delta_v;
            let ray_dir = pixel_center - camera_origin;
            let ray = Ray::new(camera_origin, ray_dir);
            let pixel_color = ray_color(&ray);
            write_color(&pixel_color);
        }
    }

    eprintln!("\rDone.                   ");
}
