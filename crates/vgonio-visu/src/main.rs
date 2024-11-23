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
    camera,
    camera::Camera,
    hit::HittableList,
    image::{linear_to_srgb, rgba_to_u32, TiledImage},
    material::{Dielectric, Lambertian, Metal},
    ray::Ray,
    sphere::Sphere,
    RenderParams,
};
use winit::event_loop::EventLoop;

#[derive(Parser, Debug)]
#[command(author, version, about)]
#[command(next_line_help = true)]
#[clap(disable_help_flag = true)]
struct Args {
    #[arg(long, action = clap::ArgAction::HelpLong)]
    help: Option<bool>,
    #[arg(short = 'w', help = "Image width in pixels.", default_value_t = 256)]
    width: u32,
    #[arg(short = 'h', help = "Image height in pixels.", default_value_t = 256)]
    height: u32,

    #[arg(
        short = 's',
        help = "Number of samples per pixel. The higher the value, the less noise in the image.",
        default_value_t = 1
    )]
    spp: u32,
    #[arg(
        long = "tsz",
        help = "Tile size (width/height) in pixels.",
        default_value_t = 32
    )]
    tile_size: u32,
    #[arg(short = 'b', default_value_t = 16)]
    max_bounces: u32,
    #[arg(long = "vfov", default_value_t = 90.0)]
    vfov_in_deg: f64,
    #[arg(
        long = "rt",
        help = "Use real-time rendering mode. The image will be presented as it is rendered.",
        default_value_t = false
    )]
    realtime: bool,
}

fn main() {
    let args = Args::parse();
    let (image_width, image_height) = (args.width, args.height);
    if args.width == 0 || args.height == 0 {
        eprintln!("Image dimensions must be positive integers.");
        return;
    }

    let camera = Camera::new(image_width, image_height, args.vfov_in_deg);
    let mut world = HittableList::default();

    let material_ground = Arc::new(Lambertian {
        albedo: Clr3::new(0.8, 0.8, 0.0),
    });
    let material_center = Arc::new(Lambertian {
        albedo: Clr3::new(0.7, 0.3, 0.3),
    });
    let material_left = Arc::new(Dielectric { ior: 1.5 });
    // let material_left = Arc::new(Metal::new(Clr3::new(0.8, 0.8, 0.8), 0.3));
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

    let mut film = TiledImage::new(image_width, image_height, args.tile_size, args.tile_size);
    let params = RenderParams {
        camera: &camera,
        world: &world,
        spp,
        max_bounces: args.max_bounces,
    };

    if args.realtime {
        let event_loop = EventLoop::new().unwrap();
        let display = pollster::block_on(vgonio_visu::display::Display::new(
            image_width,
            image_height,
            &event_loop,
        ));
        vgonio_visu::display::run(event_loop, display, render_film, &params, &mut film);
    } else {
        render_film(&params, &mut film, false);
        let mut image = image::RgbaImage::new(image_width, image_height);
        // film.write_to_image(&mut image);
        film.write_to_flat_buffer(image.as_mut());
        let filename = format!(
            "image-{}x{}-{}spp-{}bnc-{:.2}fov.png",
            image_width, image_height, spp, args.max_bounces, args.vfov_in_deg
        );
        println!("Saving image to {}", filename);
        image
            .save_with_format(filename, image::ImageFormat::Png)
            .unwrap();
    }
}

fn render_film(params: &RenderParams, film: &mut TiledImage, silent: bool) {
    use rayon::iter::ParallelIterator;
    fn render_film_inner(params: &RenderParams, film: &mut TiledImage) {
        film.par_tiles_mut().for_each(|tile| {
            tile.pixels.iter_mut().enumerate().for_each(|(i, pixel)| {
                let x = tile.x + (i % tile.w as usize) as u32;
                let y = tile.y + (i / tile.w as usize) as u32;
                *pixel = render_pixel(
                    x,
                    y,
                    params.camera,
                    params.world,
                    params.max_bounces,
                    &mut vec![Ray::empty(); params.spp as usize],
                );
            });
        });
    }

    if silent {
        render_film_inner(params, film);
    } else {
        let num_tiles = film.tiles_per_image;
        let num_done = &AtomicU32::new(0);
        let total_time = &AtomicU64::new(0);
        let max_time = &AtomicU64::new(0);
        let start_time = std::time::Instant::now();

        thread::scope(|s| {
            s.spawn(move || {
                film.par_tiles_mut().for_each(|tile| {
                    let start = std::time::Instant::now();
                    let mut rays = vec![Ray::empty(); params.spp as usize];
                    tile.pixels.iter_mut().enumerate().for_each(|(i, pixel)| {
                        let x = tile.x + (i % tile.w as usize) as u32;
                        let y = tile.y + (i / tile.w as usize) as u32;
                        *pixel = render_pixel(
                            x,
                            y,
                            params.camera,
                            params.world,
                            params.max_bounces,
                            &mut rays,
                        );
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
                }

                if n == num_tiles {
                    break;
                }

                print!(
                    "\rProgress: {}/{} tiles done, {}ms average, {}ms peek, {}ms elapsed",
                    n + 1,
                    num_tiles,
                    total_time / n as u64,
                    max_time,
                    start_time.elapsed().as_millis()
                );
                thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        println!("\nDone!");
    }
}

/// Render a single pixel in the image.
///
/// # Arguments
///
/// * `x` - The x-coordinate of the pixel.
/// * `y` - The y-coordinate of the pixel.
/// * `camera` - The camera used to generate rays.
/// * `world` - The world containing all objects.
/// * `max_bounces` - The maximum number of bounces a ray can make.
/// * `rays` - A mutable container to store rays generated by the camera.
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
        pixel_color += camera::ray_color(ray, world, 0, max_bounces) * rcp_spp;
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
