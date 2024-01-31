use clap::Parser;
use jabr::{Clr3, Vec3};
use std::{
    sync::{
        atomic::{AtomicU32, AtomicU64},
        Arc,
    },
    thread,
};
use vgonio_visu::{
    camera::{ray_color, Camera},
    hit::HittableList,
    image::{linear_to_srgb, rgba_to_u32, TiledImage},
    material::{Lambertian, Metal},
    ray::Ray,
    sphere::Sphere,
};

#[derive(Parser, Debug)]
#[command(author, version, about)]
#[command(next_line_help = true)]
#[clap(disable_help_flag = true)]
struct Args {
    /// Prints help information.
    #[arg(long, action = clap::ArgAction::HelpLong)]
    help: Option<bool>,
    /// Image width in pixels.
    #[arg(short = 'w', default_value_t = 256)]
    width: u32,
    /// Image height in pixels.
    #[arg(short = 'h', default_value_t = 256)]
    height: u32,
    /// Samples per pixel
    #[arg(short = 's', default_value_t = 1)]
    spp: u32,
    #[arg(short = 'b', default_value_t = 16)]
    max_bounces: u32,
}

fn main() {
    let args = Args::parse();
    let (image_width, image_height) = (args.width, args.height);
    if args.width == 0 || args.height == 0 {
        eprintln!("Image dimensions must be positive integers.");
        return;
    }

    let camera = Camera::new(image_width, image_height);

    // World
    let mut world = HittableList::new();

    let material_ground = Arc::new(Lambertian {
        albedo: Clr3::new(0.8, 0.8, 0.0),
    });
    let material_center = Arc::new(Lambertian {
        albedo: Clr3::new(0.7, 0.3, 0.3),
    });
    let material_left = Arc::new(Metal::new(Clr3::new(0.8, 0.8, 0.8), 0.3));
    let material_right = Arc::new(Metal::new(Clr3::new(0.8, 0.6, 0.2), 1.0));

    world.add(Arc::new(Sphere::new(
        Vec3::new(0.0, 1.0, -100.5),
        100.0,
        material_ground,
    )));
    world.add(Arc::new(Sphere::new(
        Vec3::new(0.0, 1.0, 0.0),
        0.5,
        material_center,
    )));
    world.add(Arc::new(Sphere::new(
        Vec3::new(-1.0, 1.0, 0.0),
        0.5,
        material_left,
    )));
    world.add(Arc::new(Sphere::new(
        Vec3::new(1.0, 1.0, 0.0),
        0.5,
        material_right,
    )));

    let spp = args.spp.clamp(1, u32::MAX);

    let mut film = TiledImage::new(image_width, image_height, 32, 32);

    render_film(&camera, &world, spp, args.max_bounces, &mut film);

    let mut image = image::RgbaImage::new(image_width, image_height);
    // film.write_to_image(&mut image);
    film.write_to_flat_buffer(image.as_mut());
    let filename = format!(
        "image-{}x{}-{}spp-{}bnc.png",
        image_width, image_height, spp, args.max_bounces
    );
    println!("Saving image to {}", filename);
    image
        .save_with_format(filename, image::ImageFormat::Png)
        .unwrap();
}

fn render_film(
    camera: &Camera,
    world: &HittableList,
    spp: u32,
    max_bounces: u32,
    film: &mut TiledImage,
) {
    use rayon::iter::ParallelIterator;
    let num_tiles = film.tiles_per_image;
    let num_done = &AtomicU32::new(0);
    let total_time = &AtomicU64::new(0);
    let max_time = &AtomicU64::new(0);
    let start_time = std::time::Instant::now();

    thread::scope(|s| {
        s.spawn(move || {
            film.par_tiles_mut().for_each(|tile| {
                let start = std::time::Instant::now();
                let mut rays = vec![Ray::empty(); spp as usize];
                tile.pixels.iter_mut().enumerate().for_each(|(i, pixel)| {
                    let x = tile.x + (i % tile.w as usize) as u32;
                    let y = tile.y + (i / tile.w as usize) as u32;
                    *pixel = render_pixel(x, y, camera, world, max_bounces, &mut rays);
                });
                let time_taken = start.elapsed().as_millis() as u64;

                num_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                total_time.fetch_add(time_taken, std::sync::atomic::Ordering::Relaxed);
                max_time.fetch_max(time_taken, std::sync::atomic::Ordering::Relaxed);
            });
        });

        loop {
            let n = num_done.load(std::sync::atomic::Ordering::Relaxed);
            let total_time = total_time.load(std::sync::atomic::Ordering::Relaxed);
            let max_time = max_time.load(std::sync::atomic::Ordering::Relaxed);
            if n == 0 {
                continue;
            } else if n == num_tiles {
                break;
            } else {
                print!(
                    "\rProgress: {}/{} tiles done, {}ms average, {}ms peek, {}ms elapsed",
                    n + 1,
                    num_tiles,
                    total_time / n as u64,
                    max_time,
                    start_time.elapsed().as_millis()
                );
            }
            thread::sleep(std::time::Duration::from_millis(10));
        }
    });
    println!("\nDone!");
}

fn render_pixel(
    x: u32,
    y: u32,
    camera: &Camera,
    world: &HittableList,
    max_bounces: u32,
    rays: &mut [Ray],
) -> u32 {
    let rcp_spp = 1.0 / rays.len() as f64;
    camera.generate_rays(x, y, rays);
    let mut pixel_color = Clr3::zeros();
    for ray in rays.iter_mut() {
        pixel_color += ray_color(ray, world, 0, max_bounces) * rcp_spp;
    }
    pixel_color.x = linear_to_srgb(pixel_color.x);
    pixel_color.y = linear_to_srgb(pixel_color.y);
    pixel_color.z = linear_to_srgb(pixel_color.z);
    pixel_color.clamp(0.0, 0.9999);
    let color = rgba_to_u32(
        (256.0 * pixel_color.x) as u8,
        (256.0 * pixel_color.y) as u8,
        (256.0 * pixel_color.z) as u8,
        255,
    );
    for ray in rays {
        *ray = Ray::empty();
    }
    color
}
