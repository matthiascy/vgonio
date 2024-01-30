use jabr::Pnt3;
use rand::distributions::{Distribution, Uniform};

pub fn samples_in_unit_square_2d(samples: &mut [Pnt3]) {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    for sample in samples {
        sample.x = dist.sample(&mut rng);
        sample.y = dist.sample(&mut rng);
        sample.z = 0.0;
    }
}
