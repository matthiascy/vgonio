fn main() {
    let filename = std::env::args().nth(1).expect("missing filename argument");
    let brdf = vgonio_powitacq::BrdfData::new(&std::path::Path::new(&filename));
    let wavelengths = brdf.wavelengths();
    println!("{:?}", wavelengths);
    let eval = brdf.eval(0.0, 0.0, 0.0, 0.0);
    println!("{:?}", eval);
    let n_wavelengths = brdf.n_wavelengths();
    println!("{:?}", n_wavelengths);
    assert_eq!(wavelengths.len(), n_wavelengths as usize);
}
