use clap::Parser;
use jabr::Vec3;
use vgonio_visu::{camera::Camera, hit::HittableList, sphere::Sphere};

#[derive(Parser, Debug)]
#[command(author, version, about)]
#[command(next_line_help = true)]
struct Args {
    #[arg(short = 'x', default_value_t = 256)]
    width: u32,
    #[arg(short = 'y', default_value_t = 256)]
    height: u32,
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
    world.add(Box::new(Sphere::new(Vec3::new(0.0, 1.0, 0.0), 0.5)));
    world.add(Box::new(Sphere::new(Vec3::new(0.0, 1.0, -100.5), 100.0)));

    camera.render(&world);
}
