use std::io::{BufReader, BufWriter, Cursor};

use criterion::{criterion_group, criterion_main, Criterion};
use vgn_core::io::{
    read_f32_data_samples, write_f32_data_samples_binary, CompressionScheme, FileEncoding,
};

fn payload(n: usize) -> Vec<f32> { (0..n).map(|i| (i as f32 * 0.0125).sin()).collect() }

fn bench_compression_scheme(c: &mut Criterion) {
    let samples = payload(1 << 18); // 256K samples, ~1MB uncompressed
    let n = samples.len();

    for scheme in [
        CompressionScheme::None,
        CompressionScheme::Zlib,
        CompressionScheme::Gzip,
        CompressionScheme::Lz4,
    ] {
        c.bench_function(&format!("write_{}", scheme), |b| {
            b.iter(|| {
                let mut buf = Vec::<u8>::new();
                {
                    let mut writer = BufWriter::new(&mut buf);
                    write_f32_data_samples_binary(&mut writer, scheme, &samples).unwrap();
                }
                buf
            });
        });

        // Pre-compute compressed bytes for the read bench.
        let mut compressed = Vec::<u8>::new();
        {
            let mut w = BufWriter::new(&mut compressed);
            write_f32_data_samples_binary(&mut w, scheme, &samples).unwrap();
        }

        c.bench_function(&format!("read_{}", scheme), |b| {
            b.iter(|| {
                let mut r = BufReader::new(Cursor::new(&compressed[..]));
                read_f32_data_samples(&mut r, n, FileEncoding::Binary, scheme).unwrap()
            });
        });
    }
}

criterion_group! {
    name = compression_benches;
    config = Criterion::default();
    targets = bench_compression_scheme
}
criterion_main!(compression_benches);
